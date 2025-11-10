import asyncio
import os
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional, Set, Tuple

from pydantic import Field, field_validator

from hummingbot.client.config.config_validators import validate_decimal, validate_market_trading_pair
from hummingbot.client.settings import AllConnectorSettings
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.utils import split_hb_trading_pair
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PriceType, TradeType
from hummingbot.core.event.events import BuyOrderCompletedEvent, FundingPaymentCompletedEvent, SellOrderCompletedEvent
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase


class BinancePerpSpotConfig(StrategyV2ConfigBase):
    script_file_name: str = os.path.basename(__file__)
    candles_config: List[CandlesConfig] = []
    controllers_config: List[str] = []
    markets: Dict[str, Set[str]] = {}

    perp_connector: str = Field(
        default="binance_perpetual",
        json_schema_extra={
            "prompt": lambda _: "Perpetual connector (e.g. binance_perpetual): ",
            "prompt_on_new": True,
        },
    )
    spot_connector: str = Field(
        default="mexc",
        json_schema_extra={
            "prompt": lambda _: "Spot connector (e.g. mexc): ",
            "prompt_on_new": True,
        },
    )
    trading_pair: str = Field(
        default="SOL-USDT",
        json_schema_extra={
            "prompt": lambda mi: BinancePerpSpotConfig.trading_pair_prompt(mi),
            "prompt_on_new": True,
        },
    )
    perp_leverage: int = Field(
        default=5,
        gt=0,
        json_schema_extra={
            "prompt": lambda _: "Perpetual leverage (e.g. 5): ",
            "prompt_on_new": True,
        },
    )
    order_value_quote: Decimal = Field(
        default=Decimal("10"),
        gt=Decimal("0"),
        json_schema_extra={
            "prompt": lambda mi: (
                f"Target notional value in quote asset (applies to both perp and spot;"
                f" actual fills are adjusted to the exchange min step, so treat this as an"
                f" approximate {mi.quote_asset_symbol()} amount when trading {mi.trading_pair}): "
            ),
            "prompt_on_new": True,
        },
    )

    @staticmethod
    def trading_pair_prompt(model_instance: "BinancePerpSpotConfig") -> str:
        perp_connector = model_instance.perp_connector or "binance_perpetual"
        example = AllConnectorSettings.get_example_pairs().get(perp_connector)
        example_text = f" (e.g. {example})" if example else ""
        return (
            f"Enter the trading pair to use on {perp_connector}{example_text}. "
            f"The same pair will be used on {model_instance.spot_connector or 'mexc'}."
        )

    def quote_asset_symbol(self) -> str:
        try:
            _, quote = split_hb_trading_pair(self.trading_pair.upper())
            return quote
        except Exception:
            return "quote"

    @field_validator("trading_pair", mode="after")
    @classmethod
    def validate_trading_pair(cls, value: str, info):
        trading_pair = value.upper()
        connectors = []
        perp_connector = info.data.get("perp_connector") or cls.model_fields["perp_connector"].default
        spot_connector = info.data.get("spot_connector") or cls.model_fields["spot_connector"].default
        if perp_connector:
            connectors.append(perp_connector)
        if spot_connector and spot_connector != perp_connector:
            connectors.append(spot_connector)
        for connector in connectors:
            error = validate_market_trading_pair(connector, trading_pair)
            if error is not None:
                raise ValueError(error)
        # ensure pair can be split into base-quote
        try:
            split_hb_trading_pair(trading_pair)
        except Exception as exc:
            raise ValueError(f"Invalid trading pair format '{trading_pair}'. Expected format BASE-QUOTE.") from exc
        return trading_pair

    @field_validator("order_value_quote", mode="after")
    @classmethod
    def validate_order_value(cls, value: Decimal):
        error = validate_decimal(str(value), 0, None, False)
        if error is not None:
            raise ValueError(error)
        return Decimal(str(value))


class BinancePerpMexcSpot(StrategyV2Base):
    config: BinancePerpSpotConfig
    markets: Dict[str, Set[str]] = {}

    @classmethod
    def init_markets(cls, config: BinancePerpSpotConfig):
        trading_pair = config.trading_pair.upper()
        cls.markets = {
            config.perp_connector: {trading_pair},
            config.spot_connector: {trading_pair},
        }

    def __init__(self, connectors: Dict[str, ConnectorBase], config: Optional[BinancePerpSpotConfig] = None):
        config = config or BinancePerpSpotConfig()
        super().__init__(connectors, config)
        self.perp_connector_name = self.config.perp_connector
        self.spot_connector_name = self.config.spot_connector
        self.perp_trading_pair = self.config.trading_pair.upper()
        self.spot_trading_pair = self.config.trading_pair.upper()
        base_asset, quote_asset = split_hb_trading_pair(self.perp_trading_pair)
        self.token = base_asset
        self.quote = quote_asset
        self._last_perp_price: Optional[Decimal] = None

        self.perp_connector: ConnectorBase = self.connectors[self.perp_connector_name]
        self.spot_connector: ConnectorBase = self.connectors[self.spot_connector_name]

        self._operation_task: Optional[asyncio.Task] = None
        self._execution_started = False
        self._execution_completed = False

        self._perp_order_id: Optional[str] = None
        self._spot_order_id: Optional[str] = None
        self._perp_fill_event = asyncio.Event()
        self._spot_fill_event = asyncio.Event()
        self._perp_filled_amount: Optional[Decimal] = None
        self._perp_entry_price: Optional[Decimal] = None
        self._spot_open_order_price: Optional[Decimal] = None
        self._perp_margin: Optional[Decimal] = None
        self._perp_target_amount: Optional[Decimal] = None
        self._perp_target_notional: Optional[Decimal] = None
        self._perp_open_order_price: Optional[Decimal] = None
        self._spot_filled_amount: Optional[Decimal] = None
        self._spot_entry_price: Optional[Decimal] = None
        self._perp_close_order_id: Optional[str] = None
        self._perp_close_order_price: Optional[Decimal] = None
        self._perp_close_fill_event = asyncio.Event()
        self._perp_close_filled_amount: Optional[Decimal] = None
        self._spot_close_order_id: Optional[str] = None
        self._spot_close_fill_event = asyncio.Event()
        self._spot_close_filled_amount: Optional[Decimal] = None
        self._spot_close_order_price: Optional[Decimal] = None
        self._stage: str = "not_started"
        self._closing_task: Optional[asyncio.Task] = None
        self._closing_reason: Optional[str] = None
        self._last_funding_check_ts: float = 0.0
        self._latest_funding_rate: Optional[Decimal] = None
        self._latest_funding_eta: Optional[int] = None
        self._perp_trading_rule = None
        self._spot_trading_rule = None
        self._funding_total_quote: Decimal = Decimal("0")
        self._spot_supports_market: bool = False

    def apply_initial_setting(self):
        if hasattr(self.perp_connector, "set_position_mode"):
            self.perp_connector.set_position_mode(PositionMode.ONEWAY)
        if hasattr(self.perp_connector, "set_leverage"):
            self.perp_connector.set_leverage(self.perp_trading_pair, self.config.perp_leverage)

    def on_tick(self):
        if not self.ready_to_trade:
            return
        if self._operation_task is not None and self._operation_task.done():
            self._operation_task = None
        if self._closing_task is not None and self._closing_task.done():
            self._closing_task = None
        if self._stage == "not_started":
            if self._operation_task is None:
                self._execution_started = True
                self._stage = "opening"
                self._operation_task = safe_ensure_future(self._run_single_cycle())
            return
        if self._stage == "opening":
            return
        if self._stage == "hedged":
            self._monitor_funding_and_maybe_close()
        if self._stage == "closed":
            self._execution_completed = True

    def _reset_cycle_state(self):
        self._perp_order_id = None
        self._spot_order_id = None
        self._perp_fill_event = asyncio.Event()
        self._spot_fill_event = asyncio.Event()
        self._perp_filled_amount = None
        self._perp_entry_price = None
        self._spot_open_order_price = None
        self._perp_margin = None
        self._perp_target_amount = None
        self._perp_target_notional = None
        self._perp_open_order_price = None
        self._last_perp_price = None
        self._spot_filled_amount = None
        self._spot_entry_price = None
        self._perp_close_order_id = None
        self._perp_close_order_price = None
        self._perp_close_fill_event = asyncio.Event()
        self._perp_close_filled_amount = None
        self._spot_close_order_id = None
        self._spot_close_fill_event = asyncio.Event()
        self._spot_close_filled_amount = None
        self._spot_close_order_price = None
        self._closing_task = None
        self._closing_reason = None
        self._last_funding_check_ts = 0.0
        self._latest_funding_rate = None
        self._latest_funding_eta = None

    def _monitor_funding_and_maybe_close(self):
        if self._perp_filled_amount is None or self._perp_filled_amount <= Decimal("0"):
            return
        current_time = self.current_timestamp
        if current_time - self._last_funding_check_ts < 1:
            return
        self._last_funding_check_ts = current_time
        try:
            funding_info = self.perp_connector.get_funding_info(self.perp_trading_pair)
        except Exception:
            return
        self._latest_funding_rate = funding_info.rate
        seconds_remaining = funding_info.next_funding_utc_timestamp - int(current_time)
        self._latest_funding_eta = seconds_remaining if seconds_remaining >= 0 else None
        if 0 <= seconds_remaining <= 60:
            if funding_info.rate is not None and funding_info.rate <= Decimal("0"):
                self.logger().info(
                    "资金费率在结算前 1 分钟内不为正，启动平仓流程。"
                )
                self._initiate_closing("资金费率非正")

    def _initiate_closing(self, reason: str):
        if self._stage in {"closing", "closed"}:
            return
        if self._closing_task is not None and not self._closing_task.done():
            return
        if self._perp_filled_amount is None or self._perp_filled_amount <= Decimal("0"):
            self.logger().warning("当前没有可平的永续仓位，忽略平仓请求。")
            return
        self._closing_reason = reason
        self._perp_close_fill_event = asyncio.Event()
        self._perp_close_filled_amount = None
        self._perp_close_order_id = None
        self._perp_close_order_price = None
        self._spot_close_fill_event = asyncio.Event()
        self._spot_close_filled_amount = None
        self._spot_close_order_id = None
        self._closing_task = safe_ensure_future(self._close_positions())
        self._stage = "closing"
        self.logger().info(f"开始平仓流程，原因: {reason}")

    async def _close_positions(self):
        try:
            await self._close_perp_position()
            if self._perp_close_fill_event.is_set():
                await self._close_spot_position()
                self.logger().info("平仓流程完成。")
            else:
                self.logger().warning("永续仓位未确认平仓，跳过现货平仓。")
        except Exception as ex:
            self.logger().error(f"平仓流程出现异常: {ex}", exc_info=True)
        finally:
            self._stage = "closed"
            self._execution_completed = True

    async def _close_perp_position(self):
        await self._ensure_trading_rules_ready()
        position_amount = None
        try:
            position = getattr(self.perp_connector, "get_position", lambda _: None)(self.perp_trading_pair)
            if position is not None and getattr(position, "amount", None) is not None:
                position_amount = abs(Decimal(str(position.amount)))
        except Exception:
            position_amount = None

        if position_amount is None or position_amount <= Decimal("0"):
            position_amount = self._perp_filled_amount or Decimal("0")

        if position_amount is None or position_amount <= Decimal("0"):
            self.logger().info("未持有永续仓位，跳过永续平仓。")
            return

        price = self._get_perp_market_price(TradeType.BUY) or self._last_perp_price
        if price is None or price <= Decimal("0"):
            self.logger().warning("无法获取永续平仓价格，跳过平仓。")
            return

        close_amount = self._adjust_amount_for_rule(
            amount=position_amount,
            price=price,
            rule=self._get_perp_trading_rule(),
            round_up=True,
            ensure_min_notional=False,
        )
        if close_amount is None or close_amount <= Decimal("0"):
            self.logger().warning("永续平仓数量量化为 0，跳过永续平仓。")
            return

        success = await self._execute_perp_limit_order(
            side=TradeType.BUY,
            amount=close_amount,
            position_action=PositionAction.CLOSE,
            stage="close",
        )
        if not success:
            self.logger().warning("永续平仓订单未在超时时间内完成。")

    async def _close_spot_position(self):
        await self._ensure_trading_rules_ready()
        if self._spot_filled_amount is None or self._spot_filled_amount <= Decimal("0"):
            self.logger().info("未持有现货仓位，跳过现货平仓。")
            return
        success = await self._execute_spot_order(
            side=TradeType.SELL,
            target_amount=self._spot_filled_amount,
            stage="close",
        )
        if success:
            self.logger().info("现货平仓挂单已完成。")
        else:
            self.logger().warning("现货平仓订单未在超时时间内完成。")

    async def _run_single_cycle(self):
        open_success = False
        try:
            await self._ensure_connectors_ready()
            await self._ensure_trading_rules_ready()
            self._reset_cycle_state()
            best_ask_price = self.market_data_provider.get_price_by_type(
                connector_name=self.perp_connector_name,
                trading_pair=self.perp_trading_pair,
                price_type=PriceType.BestAsk,
            )
            if best_ask_price is None or Decimal(str(best_ask_price)) <= Decimal("0"):
                self.logger().warning("无法获取永续最优卖价；终止开仓流程。")
                return
            price_decimal = Decimal(str(best_ask_price))
            self._last_perp_price = price_decimal
            perp_amount = self._calculate_perp_amount_from_quote(self.config.order_value_quote, price_decimal)
            if perp_amount is None or perp_amount <= Decimal("0"):
                self.logger().warning("永续订单数量不满足最小要求；终止开仓流程。")
                return
            perp_amount = Decimal(str(perp_amount))
            actual_notional = perp_amount * price_decimal
            self._perp_target_amount = perp_amount
            self._perp_target_notional = actual_notional

            self.logger().info(
                f"启动永续开仓追价挂单，目标数量 {perp_amount} {self.token}，目标名义价值约"
                f" {self._format_decimal(actual_notional)} {self.quote}，将一直挂在最新卖一价。"
            )

            open_success = await self._execute_perp_limit_order(
                side=TradeType.SELL,
                amount=perp_amount,
                position_action=PositionAction.OPEN,
                stage="open",
            )
            if not open_success:
                self.logger().warning("永续开仓订单未在超时时间内成交，停止开仓流程。")
                return
            if self._perp_filled_amount and self._perp_entry_price:
                self._perp_target_notional = self._perp_filled_amount * self._perp_entry_price

            filled_amount = self._perp_filled_amount or perp_amount
            spot_success = await self._execute_spot_order(
                side=TradeType.BUY,
                target_amount=filled_amount,
                stage="open",
            )
            open_success = self._perp_fill_event.is_set() and spot_success and self._spot_fill_event.is_set()
            if open_success:
                self.logger().info("开仓流程完成，等待资金费监控。")
            else:
                self.logger().warning("开仓流程未完全确认成交。")
        except Exception as ex:
            self.logger().error(f"开仓流程出现异常: {ex}", exc_info=True)
        finally:
            if open_success:
                self._stage = "hedged"
                self._execution_completed = False
            else:
                self._stage = "not_started"

    async def _ensure_connectors_ready(self):
        while not self.ready_to_trade:
            await asyncio.sleep(1)

    async def _ensure_trading_rules_ready(self):
        while self.perp_trading_pair not in getattr(self.perp_connector, "trading_rules", {}):
            await asyncio.sleep(1)
        while self.spot_trading_pair not in getattr(self.spot_connector, "trading_rules", {}):
            await asyncio.sleep(1)
        self._ensure_rule_cached(is_perp=True)
        self._ensure_rule_cached(is_perp=False)
        spot_rule = self._get_spot_trading_rule()
        rule_supports_market = getattr(spot_rule, "supports_market_orders", True)
        connector_supported_types: List[OrderType] = []
        try:
            connector_supported_types = list(self.spot_connector.supported_order_types())
        except Exception:
            connector_supported_types = []
        self._spot_supports_market = rule_supports_market and (OrderType.MARKET in connector_supported_types)

    def did_complete_funding_payment(self, funding_payment_completed_event: FundingPaymentCompletedEvent):
        if (
            funding_payment_completed_event.market == self.perp_connector_name
            and funding_payment_completed_event.trading_pair == self.perp_trading_pair
        ):
            amount = Decimal(str(funding_payment_completed_event.amount))
            rate = Decimal(str(funding_payment_completed_event.funding_rate))
            self._funding_total_quote += amount
            self.logger().info(
                "资金费结算完成：金额 %s %s，资金费率 %s，累计盈亏 %s %s",
                self._format_decimal(amount),
                self.quote,
                self._format_decimal(rate, precision=6, pct=True),
                self._format_decimal(self._funding_total_quote),
                self.quote,
            )

    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        if event.order_id == self._perp_order_id:
            base_amount_raw = getattr(event, "base_asset_amount", getattr(event, "amount", None))
            quote_amount_raw = getattr(event, "quote_asset_amount", None)
            if base_amount_raw is not None:
                base_amount = Decimal(str(base_amount_raw))
                self._perp_filled_amount = base_amount
                if quote_amount_raw is not None and base_amount > Decimal("0"):
                    quote_amount = Decimal(str(quote_amount_raw))
                    self._perp_entry_price = quote_amount / base_amount
                    leverage = Decimal(str(self.perp_connector.get_leverage(self.perp_trading_pair)))
                    if leverage > Decimal("0"):
                        self._perp_margin = (self._perp_entry_price * base_amount) / leverage
            if not self._perp_fill_event.is_set():
                self._perp_fill_event.set()
        elif event.order_id == self._spot_close_order_id:
            base_amount_raw = getattr(event, "base_asset_amount", getattr(event, "amount", None))
            quote_amount_raw = getattr(event, "quote_asset_amount", None)
            if base_amount_raw is not None and quote_amount_raw is not None:
                base_amount = Decimal(str(base_amount_raw))
                self._spot_close_filled_amount = base_amount
            if not self._spot_close_fill_event.is_set():
                self._spot_close_fill_event.set()

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        if event.order_id == self._perp_close_order_id and not self._perp_close_fill_event.is_set():
            base_amount_raw = getattr(event, "base_asset_amount", getattr(event, "amount", None))
            if base_amount_raw is not None:
                base_amount = Decimal(str(base_amount_raw))
                self._perp_close_filled_amount = base_amount
            self._perp_close_fill_event.set()
        elif event.order_id == self._spot_order_id and not self._spot_fill_event.is_set():
            base_amount_raw = getattr(event, "base_asset_amount", getattr(event, "amount", None))
            quote_amount_raw = getattr(event, "quote_asset_amount", None)
            if base_amount_raw is not None and quote_amount_raw is not None:
                base_amount = Decimal(str(base_amount_raw))
                if base_amount > Decimal("0"):
                    self._spot_filled_amount = base_amount
                    quote_amount = Decimal(str(quote_amount_raw))
                    self._spot_entry_price = quote_amount / base_amount
            self._spot_fill_event.set()

    def format_status(self) -> str:
        base_status = super().format_status()
        spot_qty = self._spot_filled_amount
        spot_unrealized_pnl = self._calculate_spot_unrealized_pnl()
        perp_unrealized_pnl = self._calculate_perp_unrealized_pnl()
        funding_rate, funding_eta = self._get_funding_info()
        leverage = self._get_perp_leverage()
        sections: List[str] = []

        def render_section(title: str, entries: List[Tuple[str, str]]) -> List[str]:
            clean_entries = [(label, value) for label, value in entries]
            if not clean_entries:
                return []
            max_label = max(len(label) for label, _ in clean_entries)
            lines = [f"  {title}:"]
            for label, value in clean_entries:
                padding = " " * (max_label - len(label))
                lines.append(f"    {label}{padding} : {value}")
            return lines

        overview = [
            ("阶段", self._stage_description()),
            ("执行开始", str(self._execution_started)),
            ("执行完成", str(self._execution_completed)),
            ("平仓原因", self._closing_reason or "-"),
        ]
        sections.extend(render_section("执行概览", overview))

        perp_section = [
            ("目标名义", f"{self._format_decimal(self._perp_target_notional or self.config.order_value_quote)} {self.quote}"),
            ("目标数量", f"{self._format_decimal(self._perp_target_amount)} {self.token}"),
            ("开仓订单", self._perp_order_id or "-"),
            ("开仓挂价", f"{self._format_decimal(self._perp_open_order_price)} {self.quote}"),
            ("永续成交", f"{self._format_decimal(self._perp_filled_amount)} {self.token}"),
            ("未实现盈亏", f"{self._format_decimal(perp_unrealized_pnl)} {self.quote}"),
            ("预计保证金", f"{self._format_decimal(self._perp_margin)} {self.quote}"),
            ("杠杆", self._format_decimal(leverage, precision=2)),
            ("平仓订单", self._perp_close_order_id or "-"),
            ("平仓挂价", f"{self._format_decimal(self._perp_close_order_price)} {self.quote}"),
        ]
        sections.extend(render_section("永续腿 (Binance)", perp_section))

        spot_section = [
            ("支持市价", "是" if self._spot_supports_market else "否"),
            ("开仓订单", self._spot_order_id or "-"),
            ("开仓挂价", f"{self._format_decimal(self._spot_open_order_price)} {self.quote}"),
            ("成交数量", f"{self._format_decimal(spot_qty)} {self.token}"),
            ("未实现盈亏", f"{self._format_decimal(spot_unrealized_pnl)} {self.quote}"),
            ("平仓订单", self._spot_close_order_id or "-"),
            ("平仓挂价", f"{self._format_decimal(self._spot_close_order_price)} {self.quote}"),
        ]
        sections.extend(render_section("现货腿 (MEXC)", spot_section))

        funding_section = [
            ("当前资金费率", self._format_decimal(funding_rate, precision=6, pct=True)),
            ("距离下次资金费", funding_eta),
            ("资金费累计", f"{self._format_decimal(self._funding_total_quote)} {self.quote}"),
        ]
        sections.extend(render_section("资金费信息", funding_section))

        if sections:
            sections.insert(0, "")
        return base_status + "\n".join(sections)

    def _calculate_spot_unrealized_pnl(self) -> Optional[Decimal]:
        if self._spot_filled_amount is None or self._spot_entry_price is None:
            return None
        current_bid = self.market_data_provider.get_price_by_type(
            connector_name=self.spot_connector_name,
            trading_pair=self.spot_trading_pair,
            price_type=PriceType.BestBid,
        )
        if current_bid is None:
            return None
        return (Decimal(str(current_bid)) - self._spot_entry_price) * self._spot_filled_amount

    def _calculate_perp_unrealized_pnl(self) -> Optional[Decimal]:
        if self._perp_filled_amount is None or self._perp_entry_price is None:
            return None
        try:
            position = getattr(self.perp_connector, "get_position", None)
            if callable(position):
                position_info = position(self.perp_trading_pair)
                if position_info is not None:
                    unrealized = getattr(position_info, "unrealized_pnl", None)
                    if unrealized is None:
                        unrealized = getattr(position_info, "unrealized_profit", None)
                    if unrealized is None:
                        unrealized = getattr(position_info, "unrealized_profit_usd", None)
                    if unrealized is not None:
                        return Decimal(str(unrealized))
        except Exception:
            self.logger().debug("读取永续未实现盈亏失败，回退到盘口估算。", exc_info=True)

        current_ask = self.market_data_provider.get_price_by_type(
            connector_name=self.perp_connector_name,
            trading_pair=self.perp_trading_pair,
            price_type=PriceType.BestAsk,
        )
        if current_ask is None:
            return None
        return (self._perp_entry_price - Decimal(str(current_ask))) * self._perp_filled_amount

    def _get_funding_info(self) -> Tuple[Optional[Decimal], str]:
        funding_rate: Optional[Decimal] = self._latest_funding_rate
        funding_eta = "-"
        eta_seconds: Optional[int] = self._latest_funding_eta
        if funding_rate is None or eta_seconds is None:
            try:
                funding_info = self.perp_connector.get_funding_info(self.perp_trading_pair)
                funding_rate = funding_info.rate
                eta_seconds = funding_info.next_funding_utc_timestamp - int(self.current_timestamp)
                self._latest_funding_rate = funding_rate
                self._latest_funding_eta = eta_seconds if eta_seconds >= 0 else None
            except Exception:
                return funding_rate, funding_eta
        if eta_seconds is not None and eta_seconds >= 0:
            hours, remainder = divmod(int(eta_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            funding_eta = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return funding_rate, funding_eta

    def _get_perp_leverage(self) -> Optional[Decimal]:
        try:
            leverage_value = self.perp_connector.get_leverage(self.perp_trading_pair)
            return Decimal(str(leverage_value))
        except Exception:
            return None

    def _calculate_perp_amount_from_quote(self, target_quote: Decimal, price: Decimal) -> Optional[Decimal]:
        if price is None or price <= Decimal("0"):
            return None
        rule = self._get_perp_trading_rule()
        amount = target_quote / price
        amount = self._adjust_amount_for_rule(
            amount=amount,
            price=price,
            rule=rule,
            round_up=True,
            ensure_min_notional=True,
        )
        return amount if amount is None or amount > Decimal("0") else None

    def _calculate_spot_amount(
        self,
        target_amount: Decimal,
        side: TradeType,
        price: Optional[Decimal] = None,
    ) -> Optional[Decimal]:
        rule = self._get_spot_trading_rule()
        effective_price = price or self._get_spot_market_price(side)
        amount = self._adjust_amount_for_rule(
            amount=target_amount,
            price=effective_price,
            rule=rule,
            round_up=side == TradeType.BUY,
            ensure_min_notional=False,
        )
        return amount if amount is None or amount > Decimal("0") else None

    async def _execute_perp_limit_order(
        self,
        side: TradeType,
        amount: Decimal,
        position_action: PositionAction,
        stage: str,
    ) -> bool:
        amount = Decimal(str(amount))
        if amount <= Decimal("0"):
            return False

        order_id_attr = "_perp_order_id" if stage == "open" else "_perp_close_order_id"
        price_attr = "_perp_open_order_price" if stage == "open" else "_perp_close_order_price"
        fill_event = self._perp_fill_event if stage == "open" else self._perp_close_fill_event

        active_order_id: Optional[str] = None
        last_order_price: Optional[Decimal] = None
        expected_stage = "opening" if stage == "open" else "closing"

        while True:
            market_price = self._get_perp_market_price(side)
            if market_price is None or market_price <= Decimal("0"):
                await asyncio.sleep(0.5)
                continue

            target_price = self._apply_price_rules(market_price, side)
            if target_price is None or target_price <= Decimal("0"):
                await asyncio.sleep(0.5)
                continue

            if active_order_id is not None:
                await self._cancel_perp_order(active_order_id)
                active_order_id = None

            fill_event.clear()
            active_order_id = self._place_perp_order(side, amount, target_price, position_action)
            setattr(self, order_id_attr, active_order_id)
            setattr(self, price_attr, target_price)
            last_order_price = target_price

            while True:
                try:
                    await asyncio.wait_for(fill_event.wait(), timeout=1.0)
                    setattr(self, order_id_attr, None)
                    return True
                except asyncio.TimeoutError:
                    if stage == "open" and self._stage != expected_stage:
                        return False
                    if stage != "open" and self._stage not in {expected_stage, "closed"}:
                        return False
                    market_price = self._get_perp_market_price(side)
                    if market_price is None or market_price <= Decimal("0"):
                        continue
                    desired_price = self._apply_price_rules(market_price, side)
                    if desired_price is None:
                        continue
                    if last_order_price is None:
                        break
                    if desired_price != last_order_price:
                        break

            await asyncio.sleep(0.2)

        if active_order_id is not None:
            await self._cancel_perp_order(active_order_id)
        setattr(self, order_id_attr, None)
        return False

    async def _execute_spot_order(
        self,
        side: TradeType,
        target_amount: Decimal,
        stage: str,
    ) -> bool:
        target_amount = Decimal(str(target_amount))
        if target_amount <= Decimal("0"):
            return False

        if self._spot_supports_market:
            return await self._execute_spot_market_order(side=side, target_amount=target_amount, stage=stage)

        return await self._execute_spot_limit_chaser_order(side=side, target_amount=target_amount, stage=stage)

    async def _execute_spot_market_order(
        self,
        side: TradeType,
        target_amount: Decimal,
        stage: str,
    ) -> bool:
        order_id_attr = "_spot_order_id" if stage == "open" else "_spot_close_order_id"
        price_attr = "_spot_open_order_price" if stage == "open" else "_spot_close_order_price"
        fill_event = self._spot_fill_event if stage == "open" else self._spot_close_fill_event

        reference_price = self._get_spot_market_price(side)
        amount = self._calculate_spot_amount(target_amount, side, reference_price)
        if amount is None or amount <= Decimal("0"):
            self.logger().warning("现货订单数量不满足最小要求；跳过现货腿。")
            return False

        fill_event.clear()
        setattr(self, price_attr, None)
        action = "买入" if side == TradeType.BUY else "卖出"
        phase = "开仓" if stage == "open" else "平仓"
        try:
            if side == TradeType.BUY:
                order_id = self.buy(
                    connector_name=self.spot_connector_name,
                    trading_pair=self.spot_trading_pair,
                    amount=amount,
                    order_type=OrderType.MARKET,
                )
            else:
                order_id = self.sell(
                    connector_name=self.spot_connector_name,
                    trading_pair=self.spot_trading_pair,
                    amount=amount,
                    order_type=OrderType.MARKET,
                )
            setattr(self, order_id_attr, order_id)
            self.logger().info(f"提交现货{phase}{action}市价单，数量 {amount} {self.token}。")
        except Exception as ex:
            self.logger().error(f"提交现货{phase}{action}市价单失败: {ex}", exc_info=True)
            setattr(self, order_id_attr, None)
            return False

        try:
            await asyncio.wait_for(fill_event.wait(), timeout=5.0)
            setattr(self, order_id_attr, None)
            return True
        except asyncio.TimeoutError:
            self.logger().warning("现货市价订单在超时时间内未确认成交。")
            setattr(self, order_id_attr, None)
            return False
        except Exception:
            self.logger().warning("等待现货市价订单完成时出现异常。", exc_info=True)
            setattr(self, order_id_attr, None)
            return False

    async def _execute_spot_limit_chaser_order(
        self,
        side: TradeType,
        target_amount: Decimal,
        stage: str,
    ) -> bool:
        order_id_attr = "_spot_order_id" if stage == "open" else "_spot_close_order_id"
        price_attr = "_spot_open_order_price" if stage == "open" else "_spot_close_order_price"
        fill_event = self._spot_fill_event if stage == "open" else self._spot_close_fill_event
        expected_stage = "opening" if stage == "open" else "closing"
        rule = self._get_spot_trading_rule()

        active_order_id: Optional[str] = None
        last_order_price: Optional[Decimal] = None
        last_amount: Optional[Decimal] = None

        while True:
            market_price = self._get_spot_market_price(side)
            if market_price is None or market_price <= Decimal("0"):
                await asyncio.sleep(0.5)
                continue

            target_price = self._apply_spot_price_rules(market_price, side)
            if target_price is None or target_price <= Decimal("0"):
                await asyncio.sleep(0.5)
                continue

            adjusted_amount = self._adjust_amount_for_rule(
                amount=target_amount,
                price=target_price,
                rule=rule,
                round_up=side == TradeType.BUY,
                ensure_min_notional=False,
            )
            if adjusted_amount is None or adjusted_amount <= Decimal("0"):
                self.logger().warning("现货订单数量不满足最小要求；跳过现货腿。")
                if active_order_id is not None:
                    await self._cancel_spot_order(active_order_id)
                    setattr(self, order_id_attr, None)
                setattr(self, price_attr, None)
                return False

            if side == TradeType.SELL and adjusted_amount > target_amount:
                adjusted_amount = self._adjust_amount_for_rule(
                    amount=target_amount,
                    price=target_price,
                    rule=rule,
                    round_up=False,
                    ensure_min_notional=False,
                )
                if adjusted_amount is None or adjusted_amount <= Decimal("0"):
                    self.logger().warning("现货平仓数量量化为 0，跳过现货腿。")
                    if active_order_id is not None:
                        await self._cancel_spot_order(active_order_id)
                        setattr(self, order_id_attr, None)
                    setattr(self, price_attr, None)
                    return False

            if active_order_id is not None:
                await self._cancel_spot_order(active_order_id)
                active_order_id = None

            fill_event.clear()
            active_order_id = self._place_spot_order(side, adjusted_amount, target_price, stage)
            setattr(self, order_id_attr, active_order_id)
            setattr(self, price_attr, target_price)
            last_order_price = target_price
            last_amount = adjusted_amount

            while True:
                try:
                    await asyncio.wait_for(fill_event.wait(), timeout=1.0)
                    setattr(self, order_id_attr, None)
                    return True
                except asyncio.TimeoutError:
                    if stage == "open" and self._stage != expected_stage:
                        if active_order_id is not None:
                            await self._cancel_spot_order(active_order_id)
                        setattr(self, order_id_attr, None)
                        setattr(self, price_attr, None)
                        return False
                    if stage != "open" and self._stage not in {expected_stage, "closed"}:
                        if active_order_id is not None:
                            await self._cancel_spot_order(active_order_id)
                        setattr(self, order_id_attr, None)
                        setattr(self, price_attr, None)
                        return False
                    market_price = self._get_spot_market_price(side)
                    if market_price is None or market_price <= Decimal("0"):
                        continue
                    desired_price = self._apply_spot_price_rules(market_price, side)
                    if desired_price is None or desired_price <= Decimal("0"):
                        continue
                    desired_amount = self._adjust_amount_for_rule(
                        amount=target_amount,
                        price=desired_price,
                        rule=rule,
                        round_up=side == TradeType.BUY,
                        ensure_min_notional=False,
                    )
                    if desired_amount is None or desired_amount <= Decimal("0"):
                        self.logger().warning("现货订单数量不满足最小要求；跳过现货腿。")
                        if active_order_id is not None:
                            await self._cancel_spot_order(active_order_id)
                        setattr(self, order_id_attr, None)
                        setattr(self, price_attr, None)
                        return False
                    if side == TradeType.SELL and desired_amount > target_amount:
                        desired_amount = self._adjust_amount_for_rule(
                            amount=target_amount,
                            price=desired_price,
                            rule=rule,
                            round_up=False,
                            ensure_min_notional=False,
                        )
                        if desired_amount is None or desired_amount <= Decimal("0"):
                            self.logger().warning("现货平仓数量量化为 0，跳过现货腿。")
                            if active_order_id is not None:
                                await self._cancel_spot_order(active_order_id)
                            setattr(self, order_id_attr, None)
                            setattr(self, price_attr, None)
                            return False
                    if desired_price != last_order_price or desired_amount != last_amount:
                        break
            await asyncio.sleep(0.2)

    def _apply_spot_price_rules(self, price: Decimal, side: TradeType) -> Optional[Decimal]:
        rule = self._get_spot_trading_rule()
        if price is None or price <= Decimal("0"):
            return None
        step = self._decimal_from_rule_value(getattr(rule, "min_price_increment", None))
        if step and step > Decimal("0"):
            rounding = ROUND_CEILING if side == TradeType.BUY else ROUND_FLOOR
            price = (price / step).to_integral_value(rounding=rounding) * step
        return price if price > Decimal("0") else None

    def _place_spot_order(
        self,
        side: TradeType,
        amount: Decimal,
        price: Decimal,
        stage: str,
    ) -> str:
        action = "买入" if side == TradeType.BUY else "卖出"
        phase = "开仓" if stage == "open" else "平仓"
        if side == TradeType.BUY:
            order_id = self.buy(
                connector_name=self.spot_connector_name,
                trading_pair=self.spot_trading_pair,
                amount=amount,
                order_type=OrderType.LIMIT,
                price=price,
            )
        else:
            order_id = self.sell(
                connector_name=self.spot_connector_name,
                trading_pair=self.spot_trading_pair,
                amount=amount,
                order_type=OrderType.LIMIT,
                price=price,
            )
        self.logger().info(
            f"提交现货{phase}{action}挂单，数量 {amount} {self.token}，价格 {price}。"
        )
        return order_id

    async def _cancel_spot_order(self, order_id: str):
        try:
            self.cancel(self.spot_connector_name, self.spot_trading_pair, order_id)
        except Exception:
            self.logger().warning(f"取消现货挂单失败，订单编号 {order_id}。", exc_info=True)
        await asyncio.sleep(0.1)

    def _ensure_rule_cached(self, is_perp: bool):
        rules = self.perp_connector.trading_rules if is_perp else self.spot_connector.trading_rules
        trading_pair = self.perp_trading_pair if is_perp else self.spot_trading_pair
        if trading_pair not in rules:
            raise ValueError(f"Trading rule for {trading_pair} not available yet.")
        if is_perp and self._perp_trading_rule is None:
            self._perp_trading_rule = rules[trading_pair]
        if not is_perp and self._spot_trading_rule is None:
            self._spot_trading_rule = rules[trading_pair]

    def _get_perp_trading_rule(self):
        self._ensure_rule_cached(is_perp=True)
        return self._perp_trading_rule

    def _get_spot_trading_rule(self):
        self._ensure_rule_cached(is_perp=False)
        return self._spot_trading_rule

    @staticmethod
    def _decimal_from_rule_value(value: Optional[Decimal]) -> Optional[Decimal]:
        if value is None:
            return None
        try:
            decimal_value = Decimal(str(value))
        except Exception:
            return None
        return decimal_value if decimal_value > Decimal("0") else None

    def _adjust_amount_for_rule(
        self,
        amount: Decimal,
        price: Optional[Decimal],
        rule,
        round_up: bool,
        ensure_min_notional: bool,
    ) -> Optional[Decimal]:
        if amount is None:
            return None
        amount = Decimal(str(amount))
        step = self._decimal_from_rule_value(getattr(rule, "min_base_amount_increment", None))
        min_size = self._decimal_from_rule_value(getattr(rule, "min_order_size", None))
        rounding_mode = ROUND_CEILING if round_up else ROUND_FLOOR
        if step and step > Decimal("0"):
            amount = (amount / step).to_integral_value(rounding=rounding_mode) * step
        if min_size and amount < min_size:
            amount = min_size if round_up else Decimal("0")
        if ensure_min_notional and price is not None and price > Decimal("0"):
            min_notional = (
                self._decimal_from_rule_value(getattr(rule, "min_notional_size", None))
                or self._decimal_from_rule_value(getattr(rule, "min_order_value", None))
            )
            if min_notional and amount * price < min_notional:
                if round_up:
                    required = min_notional / price
                    if step and step > Decimal("0"):
                        amount = (required / step).to_integral_value(rounding=ROUND_CEILING) * step
                    else:
                        amount = required
                else:
                    amount = Decimal("0")
        return amount if amount > Decimal("0") else None

    def _get_perp_market_price(self, side: TradeType) -> Optional[Decimal]:
        price_type = PriceType.BestBid if side == TradeType.BUY else PriceType.BestAsk
        value = self.market_data_provider.get_price_by_type(
            connector_name=self.perp_connector_name,
            trading_pair=self.perp_trading_pair,
            price_type=price_type,
        )
        if value is None:
            return self._last_perp_price
        price = Decimal(str(value))
        if price > Decimal("0"):
            self._last_perp_price = price
            return price
        return self._last_perp_price

    def _get_spot_market_price(self, side: Optional[TradeType] = None) -> Optional[Decimal]:
        price_type = PriceType.BestAsk
        if side == TradeType.BUY:
            price_type = PriceType.BestAsk
        elif side == TradeType.SELL:
            price_type = PriceType.BestBid
        value = self.market_data_provider.get_price_by_type(
            connector_name=self.spot_connector_name,
            trading_pair=self.spot_trading_pair,
            price_type=price_type,
        )
        if value is None:
            return None
        price = Decimal(str(value))
        return price if price > Decimal("0") else None

    def _apply_price_rules(self, price: Decimal, side: TradeType) -> Optional[Decimal]:
        rule = self._get_perp_trading_rule()
        if price is None or price <= Decimal("0"):
            return None
        step = self._decimal_from_rule_value(getattr(rule, "min_price_increment", None))
        if step and step > Decimal("0"):
            rounding = ROUND_FLOOR if side == TradeType.BUY else ROUND_CEILING
            price = (price / step).to_integral_value(rounding=rounding) * step
        return price if price > Decimal("0") else None

    def _place_perp_order(
        self,
        side: TradeType,
        amount: Decimal,
        price: Decimal,
        position_action: PositionAction,
    ) -> str:
        if side == TradeType.SELL:
            order_id = self.sell(
                connector_name=self.perp_connector_name,
                trading_pair=self.perp_trading_pair,
                amount=amount,
                order_type=OrderType.LIMIT_MAKER,
                price=price,
                position_action=position_action,
            )
        else:
            order_id = self.buy(
                connector_name=self.perp_connector_name,
                trading_pair=self.perp_trading_pair,
                amount=amount,
                order_type=OrderType.LIMIT_MAKER,
                price=price,
                position_action=position_action,
            )
        self.logger().info(
            f"提交永续{'开仓' if side == TradeType.SELL else '平仓'}挂单，数量 {amount} {self.token}，价格 {price}。"
        )
        return order_id

    async def _cancel_perp_order(self, order_id: str):
        try:
            self.cancel(self.perp_connector_name, self.perp_trading_pair, order_id)
        except Exception:
            self.logger().warning(f"取消永续挂单失败，订单编号 {order_id}。", exc_info=True)
        await asyncio.sleep(0.1)

    def _stage_description(self) -> str:
        mapping = {
            "not_started": "待开仓",
            "opening": "开仓中",
            "hedged": "持仓中",
            "closing": "平仓中",
            "closed": "已平仓",
        }
        return mapping.get(self._stage, self._stage)

    @staticmethod
    def _format_decimal(
        value: Optional[Decimal],
        precision: int = 4,
        pct: bool = False,
    ) -> str:
        if value is None:
            return "-"
        display_value = value * Decimal("100") if pct else value
        quantize_str = Decimal(f"1e-{precision}")
        try:
            formatted = display_value.quantize(quantize_str, rounding=ROUND_HALF_UP)
        except Exception:
            formatted = Decimal(display_value)
        suffix = "%" if pct else ""
        return f"{formatted}{suffix}"
