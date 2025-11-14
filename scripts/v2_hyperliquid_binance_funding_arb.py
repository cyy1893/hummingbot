import asyncio
import json
import os
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Dict, List, Literal, Optional, Set
from urllib.error import URLError
from urllib.request import urlopen

from pydantic import Field, field_validator, model_validator

from hummingbot.client.config.config_validators import validate_decimal, validate_market_trading_pair
from hummingbot.client.settings import AllConnectorSettings
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.utils import split_hb_trading_pair
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PriceType, TradeType
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    FundingPaymentCompletedEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    SellOrderCompletedEvent,
)
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase

FUNDING_WINDOW_SECONDS = 60
SECONDS_PER_HOUR = 3600
DEFAULT_FUNDING_INTERVAL_HOURS = Decimal("1")
CONNECTOR_FUNDING_INTERVAL_HOURS: Dict[str, Decimal] = {
    "hyperliquid_perpetual": Decimal("1"),
    "hyperliquid_perpetual_testnet": Decimal("1"),
    "binance_perpetual": Decimal("8"),
    "binance_perpetual_testnet": Decimal("8"),
}
BINANCE_INTERVAL_REFRESH_SECONDS = 1800
BINANCE_ALLOWED_INTERVALS_HOURS = {Decimal("1"), Decimal("2"), Decimal("4"), Decimal("8")}


class HyperliquidBinancePerpConfig(StrategyV2ConfigBase):
    script_file_name: str = os.path.basename(__file__)
    candles_config: List[CandlesConfig] = []
    controllers_config: List[str] = []
    markets: Dict[str, Set[str]] = {}

    hyperliquid_connector: str = Field(
        default="hyperliquid_perpetual",
        json_schema_extra={
            "prompt": lambda _: "Hyperliquid腿使用的连接器（例如 hyperliquid_perpetual）：",
            "prompt_on_new": True,
        },
    )
    binance_connector: str = Field(
        default="binance_perpetual",
        json_schema_extra={
            "prompt": lambda _: "Binance腿使用的连接器（例如 binance_perpetual）：",
            "prompt_on_new": True,
        },
    )
    hyperliquid_trading_pair: str = Field(
        default="SOL-USDC",
        json_schema_extra={
            "prompt": lambda mi: HyperliquidBinancePerpConfig.trading_pair_prompt(mi, leg="hyperliquid"),
            "prompt_on_new": True,
        },
    )
    binance_trading_pair: str = Field(
        default="SOL-USDT",
        json_schema_extra={
            "prompt": lambda mi: HyperliquidBinancePerpConfig.trading_pair_prompt(mi, leg="binance"),
            "prompt_on_new": True,
        },
    )
    leverage: int = Field(
        default=5,
        gt=0,
        json_schema_extra={
            "prompt": lambda _: "两个连接器共同使用的永续杠杆（例如 5）：",
            "prompt_on_new": True,
        },
    )
    order_value_quote: Decimal = Field(
        default=Decimal("10"),
        gt=Decimal("0"),
        json_schema_extra={
            "prompt": lambda _: "每条腿的目标保证金，美元计价（例如 10）：",
            "prompt_on_new": True,
        },
    )
    min_entry_spread: Decimal = Field(
        default=Decimal("5"),
        ge=Decimal("0"),
        json_schema_extra={
            "prompt": lambda _: "开仓所需的空多价差，单位为基点（例如 5 表示 5 个基点）：",
            "prompt_on_new": False,
        },
    )
    hyperliquid_position: Literal["long", "short"] = Field(
        default="long",
        json_schema_extra={
            "prompt": lambda _: "Hyperliquid 持仓方向（long/short）：",
            "prompt_on_new": True,
        },
    )
    binance_position: Literal["long", "short"] = Field(
        default="short",
        json_schema_extra={
            "prompt": lambda _: "Binance 持仓方向（long/short）：",
            "prompt_on_new": True,
        },
    )

    @staticmethod
    def trading_pair_prompt(model_instance: "HyperliquidBinancePerpConfig", leg: str) -> str:
        if leg == "hyperliquid":
            connector_name = model_instance.hyperliquid_connector or "hyperliquid_perpetual"
        else:
            connector_name = model_instance.binance_connector or "binance_perpetual"
        example = AllConnectorSettings.get_example_pairs().get(connector_name)
        example_text = f"（例如 {example}）" if example else ""
        return f"请输入 {connector_name} 的交易对{example_text}："

    @field_validator("hyperliquid_trading_pair", "binance_trading_pair", mode="after")
    @classmethod
    def validate_trading_pair(cls, value: str, info):
        connector_field = "hyperliquid_connector" if info.field_name == "hyperliquid_trading_pair" else "binance_connector"
        connector = info.data.get(connector_field) or cls.model_fields[connector_field].default
        pair = value.upper()
        error = validate_market_trading_pair(connector, pair)
        if error is not None:
            raise ValueError(error)
        try:
            split_hb_trading_pair(pair)
        except Exception as exc:
            raise ValueError(f"Invalid trading pair format '{value}'. Expected BASE-QUOTE.") from exc
        return pair

    @field_validator("order_value_quote", mode="after")
    @classmethod
    def validate_order_value(cls, value: Decimal):
        error = validate_decimal(str(value), 0, None, False)
        if error is not None:
            raise ValueError(error)
        return Decimal(str(value))

    @field_validator("hyperliquid_position", "binance_position", mode="after")
    @classmethod
    def normalize_position(cls, value: str):
        value_lower = value.lower()
        if value_lower not in {"long", "short"}:
            raise ValueError("Position direction must be 'long' or 'short'.")
        return value_lower

    @model_validator(mode="after")
    def validate_opposite_positions(self):
        if self.hyperliquid_position == self.binance_position:
            fields_set = getattr(self, "model_fields_set", set())
            hyper_set = "hyperliquid_position" in fields_set
            binance_set = "binance_position" in fields_set
            if not (hyper_set and binance_set):
                # Defer validation until both prompts have been answered explicitly.
                return self
            raise ValueError("Hyperliquid and Binance positions must be opposite.")
        return self


class HyperliquidBinancePerpArb(StrategyV2Base):
    markets: Dict[str, Set[str]] = {}

    @classmethod
    def init_markets(cls, config: HyperliquidBinancePerpConfig):
        cls.markets = {
            config.hyperliquid_connector: {config.hyperliquid_trading_pair.upper()},
            config.binance_connector: {config.binance_trading_pair.upper()},
        }

    def __init__(
        self,
        connectors: Dict[str, ConnectorBase],
        config: Optional[HyperliquidBinancePerpConfig] = None,
    ):
        config = config or HyperliquidBinancePerpConfig()
        super().__init__(connectors, config)
        self.config = config
        self.hyperliquid_connector_name = self.config.hyperliquid_connector
        self.binance_connector_name = self.config.binance_connector
        self.hyperliquid_trading_pair = self.config.hyperliquid_trading_pair.upper()
        self.binance_trading_pair = self.config.binance_trading_pair.upper()

        hyper_base, _ = split_hb_trading_pair(self.hyperliquid_trading_pair)
        binance_base, _ = split_hb_trading_pair(self.binance_trading_pair)
        if hyper_base != binance_base:
            self.logger().warning(
                "Configured trading pairs use different base assets (%s vs %s). Exposure may not remain hedged.",
                hyper_base,
                binance_base,
            )
        self.base_asset = hyper_base

        self.hyperliquid_connector: ConnectorBase = self.connectors[self.hyperliquid_connector_name]
        self.binance_connector: ConnectorBase = self.connectors[self.binance_connector_name]

        self.hyperliquid_open_side = TradeType.BUY if self.config.hyperliquid_position == "long" else TradeType.SELL
        self.hyperliquid_close_side = TradeType.SELL if self.hyperliquid_open_side == TradeType.BUY else TradeType.BUY
        self.binance_open_side = TradeType.BUY if self.config.binance_position == "long" else TradeType.SELL
        self.binance_close_side = TradeType.SELL if self.binance_open_side == TradeType.BUY else TradeType.BUY
        self._min_entry_spread = Decimal(str(self.config.min_entry_spread)) / Decimal("10000")
        self._short_open_side: TradeType = TradeType.SELL
        self._long_open_side: TradeType = TradeType.BUY

        self._assign_leg_roles()

        self._operation_task: Optional[asyncio.Task] = None
        self._closing_task: Optional[asyncio.Task] = None
        self._execution_started: bool = False
        self._execution_completed: bool = False
        self._stage: str = "not_started"
        self._closing_recovery: bool = False

        # Short leg state (connector assigned dynamically based on config)
        self._short_order_id: Optional[str] = None
        self._short_open_order_price: Optional[Decimal] = None
        self._short_fill_event: asyncio.Event = asyncio.Event()
        self._short_filled_amount: Optional[Decimal] = None
        self._short_entry_price: Optional[Decimal] = None
        self._short_target_amount: Optional[Decimal] = None
        self._short_target_notional: Optional[Decimal] = None
        self._short_close_order_id: Optional[str] = None
        self._short_close_order_price: Optional[Decimal] = None
        self._short_close_fill_event: asyncio.Event = asyncio.Event()
        self._short_close_filled_amount: Optional[Decimal] = None

        # Long leg state
        self._long_order_id: Optional[str] = None
        self._long_open_order_price: Optional[Decimal] = None
        self._long_fill_event: asyncio.Event = asyncio.Event()
        self._long_filled_amount: Optional[Decimal] = None
        self._long_entry_price: Optional[Decimal] = None
        self._long_target_amount: Optional[Decimal] = None
        self._long_target_notional: Optional[Decimal] = None
        self._long_close_order_id: Optional[str] = None
        self._long_close_order_price: Optional[Decimal] = None
        self._long_close_fill_event: asyncio.Event = asyncio.Event()
        self._long_close_filled_amount: Optional[Decimal] = None

        # Cached trading rules and prices
        self._short_trading_rule = None
        self._long_trading_rule = None
        self._last_short_price: Optional[Decimal] = None
        self._last_long_price: Optional[Decimal] = None

        # Funding monitoring
        self._funding_total_quote: Decimal = Decimal("0")
        self._last_funding_check_ts: float = 0.0
        self._latest_funding_rate: Optional[Decimal] = None
        self._latest_funding_eta: Optional[int] = None
        self._latest_binance_funding_rate: Optional[Decimal] = None
        self._latest_binance_funding_eta: Optional[int] = None

        # Order ownership map for event handling
        self._order_owner: Dict[str, tuple[str, str]] = {}
        self._strategy_cancelled_orders: Set[str] = set()
        self._connector_interval_overrides: Dict[str, Decimal] = {}
        self._binance_interval_last_refresh: float = 0.0
        self._binance_last_next_funding_ts: Optional[int] = None

    def apply_initial_setting(self):
        for connector_name, trading_pair in [
            (self.hyperliquid_connector_name, self.hyperliquid_trading_pair),
            (self.binance_connector_name, self.binance_trading_pair),
        ]:
            connector = self.connectors[connector_name]
            try:
                if hasattr(connector, "set_position_mode"):
                    connector.set_position_mode(PositionMode.ONEWAY)
            except Exception:
                self.logger().warning("Failed to set position mode for %s %s", connector_name, trading_pair)
            try:
                if hasattr(connector, "set_leverage"):
                    connector.set_leverage(trading_pair, self.config.leverage)
            except Exception:
                self.logger().warning("Failed to set leverage for %s %s", connector_name, trading_pair)

    def _assign_leg_roles(self):
        if self.hyperliquid_open_side == TradeType.SELL:
            self.short_connector_name = self.hyperliquid_connector_name
            self.short_trading_pair = self.hyperliquid_trading_pair
            self.short_connector = self.hyperliquid_connector
            self.long_connector_name = self.binance_connector_name
            self.long_trading_pair = self.binance_trading_pair
            self.long_connector = self.binance_connector
            self._short_open_side = self.hyperliquid_open_side
            self._long_open_side = self.binance_open_side
        else:
            self.short_connector_name = self.binance_connector_name
            self.short_trading_pair = self.binance_trading_pair
            self.short_connector = self.binance_connector
            self.long_connector_name = self.hyperliquid_connector_name
            self.long_trading_pair = self.hyperliquid_trading_pair
            self.long_connector = self.hyperliquid_connector
            self._short_open_side = self.binance_open_side
            self._long_open_side = self.hyperliquid_open_side

    def _leg_for_connector_name(self, connector_name: str) -> str:
        return "short" if connector_name == self.short_connector_name else "long"

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

    async def _run_single_cycle(self) -> None:
        open_success = False
        try:
            await self._ensure_connectors_ready()
            await self._ensure_trading_rules_ready()
            self._reset_cycle_state()

            hyper_price = self._get_market_price(
                self.hyperliquid_connector_name,
                self.hyperliquid_trading_pair,
                self.hyperliquid_open_side,
            )
            binance_price = self._get_market_price(
                self.binance_connector_name,
                self.binance_trading_pair,
                self.binance_open_side,
            )
            if hyper_price is None or hyper_price <= Decimal("0"):
                self.logger().warning("Unable to fetch price on %s when opening.", self.hyperliquid_connector_name)
                return
            if binance_price is None or binance_price <= Decimal("0"):
                self.logger().warning("Unable to fetch price on %s when opening.", self.binance_connector_name)
                return

            margin_target = Decimal(str(self.config.order_value_quote))
            leverage_decimal = Decimal(str(self.config.leverage))
            hyper_amount = self._calculate_amount_from_margin(
                connector_name=self.hyperliquid_connector_name,
                trading_pair=self.hyperliquid_trading_pair,
                side=self.hyperliquid_open_side,
                target_margin=margin_target,
                price=hyper_price,
                leverage=leverage_decimal,
            )
            binance_amount = self._calculate_amount_from_margin(
                connector_name=self.binance_connector_name,
                trading_pair=self.binance_trading_pair,
                side=self.binance_open_side,
                target_margin=margin_target,
                price=binance_price,
                leverage=leverage_decimal,
            )

            if hyper_amount is None or hyper_amount <= Decimal("0"):
                self.logger().warning("Hyperliquid leg amount below trading rule minimum; aborting open.")
                return
            if binance_amount is None or binance_amount <= Decimal("0"):
                self.logger().warning("Binance leg amount below trading rule minimum; aborting open.")
                return

            hyper_amount_dec = Decimal(str(hyper_amount))
            binance_amount_dec = Decimal(str(binance_amount))
            connector_amounts = {
                self.hyperliquid_connector_name: hyper_amount_dec,
                self.binance_connector_name: binance_amount_dec,
            }
            prices_by_connector = {
                self.hyperliquid_connector_name: hyper_price,
                self.binance_connector_name: binance_price,
            }

            self._short_target_amount = connector_amounts[self.short_connector_name]
            self._long_target_amount = connector_amounts[self.long_connector_name]
            notional_target = margin_target * leverage_decimal
            self._short_target_notional = notional_target
            self._long_target_notional = notional_target

            short_price = prices_by_connector[self.short_connector_name]
            long_price = prices_by_connector[self.long_connector_name]
            self.logger().info(
                "Launching concurrent hedged open: short %s %s @ %s, long %s %s @ %s.",
                self._format_decimal(self._short_target_amount),
                self.base_asset,
                self._format_decimal(short_price),
                self._format_decimal(self._long_target_amount),
                self.base_asset,
                self._format_decimal(long_price),
            )

            open_results = await asyncio.gather(
                self._execute_limit_order(
                    connector_name=self.hyperliquid_connector_name,
                    trading_pair=self.hyperliquid_trading_pair,
                    side=self.hyperliquid_open_side,
                    amount=connector_amounts[self.hyperliquid_connector_name],
                    position_action=PositionAction.OPEN,
                    leg=self._leg_for_connector_name(self.hyperliquid_connector_name),
                    stage="open",
                ),
                self._execute_limit_order(
                    connector_name=self.binance_connector_name,
                    trading_pair=self.binance_trading_pair,
                    side=self.binance_open_side,
                    amount=connector_amounts[self.binance_connector_name],
                    position_action=PositionAction.OPEN,
                    leg=self._leg_for_connector_name(self.binance_connector_name),
                    stage="open",
                ),
            )
            open_success = all(open_results)
            if open_success:
                self.logger().info("Hedge established; awaiting funding accrual.")
        except Exception as exc:
            self.logger().error("Exception during opening cycle: %s", exc, exc_info=True)
        finally:
            if open_success and self._short_fill_event.is_set() and self._long_fill_event.is_set():
                self._stage = "hedged"
                self._execution_completed = False
            else:
                await self._cancel_active_orders()
                if self._has_open_exposure():
                    self.logger().warning("Opening leg mismatch detected; flattening before next attempt.")
                    started = self._initiate_closing("partial fill recovery", resume_open=True)
                    if not started:
                        self._stage = "not_started"
                else:
                    self._stage = "not_started"

    async def _ensure_connectors_ready(self):
        while not self.ready_to_trade:
            await asyncio.sleep(1)

    async def _ensure_trading_rules_ready(self):
        while self.hyperliquid_trading_pair not in getattr(self.hyperliquid_connector, "trading_rules", {}):
            await asyncio.sleep(1)
        while self.binance_trading_pair not in getattr(self.binance_connector, "trading_rules", {}):
            await asyncio.sleep(1)
        hyper_rule = self.hyperliquid_connector.trading_rules[self.hyperliquid_trading_pair]
        binance_rule = self.binance_connector.trading_rules[self.binance_trading_pair]
        if self.short_connector_name == self.hyperliquid_connector_name:
            self._short_trading_rule = hyper_rule
            self._long_trading_rule = binance_rule
        else:
            self._short_trading_rule = binance_rule
            self._long_trading_rule = hyper_rule

    def _reset_cycle_state(self):
        self._short_order_id = None
        self._short_open_order_price = None
        self._short_fill_event = asyncio.Event()
        self._short_filled_amount = None
        self._short_entry_price = None
        self._short_target_amount = None
        self._short_target_notional = None
        self._short_close_order_id = None
        self._short_close_order_price = None
        self._short_close_fill_event = asyncio.Event()
        self._short_close_filled_amount = None

        self._long_order_id = None
        self._long_open_order_price = None
        self._long_fill_event = asyncio.Event()
        self._long_filled_amount = None
        self._long_entry_price = None
        self._long_target_amount = None
        self._long_target_notional = None
        self._long_close_order_id = None
        self._long_close_order_price = None
        self._long_close_fill_event = asyncio.Event()
        self._long_close_filled_amount = None

        self._order_owner.clear()
        self._funding_total_quote = Decimal("0")
        self._strategy_cancelled_orders.clear()

    def _has_open_exposure(self) -> bool:
        for filled_amount in (self._short_filled_amount, self._long_filled_amount):
            if filled_amount is not None and filled_amount > Decimal("0"):
                return True
        return False

    async def _execute_limit_order(
        self,
        connector_name: str,
        trading_pair: str,
        side: TradeType,
        amount: Decimal,
        position_action: PositionAction,
        leg: str,
        stage: str,
    ) -> bool:
        amount = Decimal(str(amount))
        if amount <= Decimal("0"):
            return False

        rule = self._short_trading_rule if leg == "short" else self._long_trading_rule
        order_id_attr, price_attr, fill_event = self._leg_attributes(leg, stage)
        active_order_id: Optional[str] = getattr(self, order_id_attr)
        expected_stage = "opening" if stage == "open" else "closing"
        last_order_price: Optional[Decimal] = None

        while True:
            market_price = self._get_market_price(connector_name, trading_pair, side)
            if market_price is None or market_price <= Decimal("0"):
                await asyncio.sleep(0.5)
                continue

            target_price = self._apply_price_rules(market_price, rule, side)
            if target_price is None or target_price <= Decimal("0"):
                await asyncio.sleep(0.5)
                continue

            if stage == "open" and not self._meets_entry_spread_requirement(leg, stage, target_price):
                await asyncio.sleep(0.5)
                continue

            if active_order_id is not None:
                await self._cancel_order(connector_name, trading_pair, active_order_id)
                active_order_id = None

            fill_event.clear()
            order_id = self._place_limit_order(
                connector_name=connector_name,
                trading_pair=trading_pair,
                side=side,
                amount=amount,
                price=target_price,
                position_action=position_action,
            )
            setattr(self, order_id_attr, order_id)
            setattr(self, price_attr, target_price)
            self._order_owner[order_id] = (leg, stage)
            active_order_id = order_id
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
                    market_price = self._get_market_price(connector_name, trading_pair, side)
                    if market_price is None or market_price <= Decimal("0"):
                        continue
                    desired_price = self._apply_price_rules(market_price, rule, side)
                    if desired_price is None:
                        continue
                    if last_order_price is None:
                        break
                    if desired_price != last_order_price:
                        break

            await asyncio.sleep(0.2)

        return False

    def _leg_attributes(self, leg: str, stage: str):
        if leg == "short":
            if stage == "open":
                return "_short_order_id", "_short_open_order_price", self._short_fill_event
            else:
                return "_short_close_order_id", "_short_close_order_price", self._short_close_fill_event
        if leg == "long":
            if stage == "open":
                return "_long_order_id", "_long_open_order_price", self._long_fill_event
            else:
                return "_long_close_order_id", "_long_close_order_price", self._long_close_fill_event
        raise ValueError(f"Unknown leg {leg}")

    def _connector_details_for_leg(self, leg: str) -> tuple[str, str]:
        if leg == "short":
            return self.short_connector_name, self.short_trading_pair
        if leg == "long":
            return self.long_connector_name, self.long_trading_pair
        raise ValueError(f"Unknown leg {leg}")

    def _place_limit_order(
        self,
        connector_name: str,
        trading_pair: str,
        side: TradeType,
        amount: Decimal,
        price: Decimal,
        position_action: PositionAction,
    ) -> str:
        if side == TradeType.BUY:
            order_id = self.buy(
                connector_name=connector_name,
                trading_pair=trading_pair,
                amount=amount,
                order_type=OrderType.LIMIT_MAKER,
                price=price,
                position_action=position_action,
            )
        else:
            order_id = self.sell(
                connector_name=connector_name,
                trading_pair=trading_pair,
                amount=amount,
                order_type=OrderType.LIMIT_MAKER,
                price=price,
                position_action=position_action,
            )
        self.logger().debug(
            "Placed %s order %s on %s %s price %s amount %s",
            "buy" if side == TradeType.BUY else "sell",
            order_id,
            connector_name,
            trading_pair,
            price,
            amount,
        )
        return order_id

    async def _cancel_order(self, connector_name: str, trading_pair: str, order_id: str):
        if order_id is None:
            return
        self._strategy_cancelled_orders.add(order_id)
        try:
            self.cancel(connector_name, trading_pair, order_id)
        except Exception:
            self._strategy_cancelled_orders.discard(order_id)
            self.logger().warning("Failed to cancel order %s on %s %s", order_id, connector_name, trading_pair, exc_info=True)
        await asyncio.sleep(0.1)

    async def _cancel_active_orders(self):
        tasks = []
        cancel_targets = [
            ("short", self._short_order_id),
            ("long", self._long_order_id),
            ("short", self._short_close_order_id),
            ("long", self._long_close_order_id),
        ]
        for leg, order_id in cancel_targets:
            if order_id:
                connector_name, trading_pair = self._connector_details_for_leg(leg)
                tasks.append(self._cancel_order(connector_name, trading_pair, order_id))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _funding_payout_value(
        self,
        rate: Optional[Decimal],
        open_side: TradeType,
        multiplier: Decimal = Decimal("1"),
    ) -> Optional[Decimal]:
        if rate is None:
            return None
        signed_rate = rate if open_side == TradeType.SELL else -rate
        return signed_rate * multiplier

    def _seconds_to_next_hour(self, timestamp: float) -> int:
        seconds_into_hour = int(timestamp) % SECONDS_PER_HOUR
        seconds_remaining = SECONDS_PER_HOUR - seconds_into_hour
        return seconds_remaining

    def _settlement_hours_for_connector(self, connector_name: str) -> Decimal:
        connector_key = connector_name.lower()
        override = self._connector_interval_overrides.get(connector_key)
        if override is not None and override > Decimal("0"):
            return override
        for name, hours in CONNECTOR_FUNDING_INTERVAL_HOURS.items():
            if name in connector_key:
                return hours
        return DEFAULT_FUNDING_INTERVAL_HOURS

    def _hourly_rate_for_connector(self, rate: Optional[Decimal], connector_name: str) -> Optional[Decimal]:
        if rate is None:
            return None
        settlement_hours = self._settlement_hours_for_connector(connector_name)
        if settlement_hours <= Decimal("0"):
            return rate
        return rate / settlement_hours

    def _normalize_binance_interval(self, hours: Decimal) -> Decimal:
        if not BINANCE_ALLOWED_INTERVALS_HOURS:
            return hours
        closest = min(BINANCE_ALLOWED_INTERVALS_HOURS, key=lambda candidate: abs(candidate - hours))
        return closest

    def _set_connector_interval_override(self, connector_name: str, hours: Decimal):
        if hours is None or hours <= Decimal("0"):
            return
        normalized_name = connector_name.lower()
        if "binance_perpetual" in normalized_name:
            hours = self._normalize_binance_interval(hours)
        self._connector_interval_overrides[normalized_name] = hours

    def _binance_symbol(self) -> str:
        base, quote = split_hb_trading_pair(self.binance_trading_pair)
        return f"{base}{quote}"

    def _fetch_binance_interval_from_api(self) -> Optional[Decimal]:
        symbol = self._binance_symbol()
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=2"
        try:
            with urlopen(url, timeout=5) as response:
                payload = response.read().decode("utf-8")
            data = json.loads(payload)
        except asyncio.CancelledError:
            raise
        except URLError:
            self.logger().warning(
                "Network error fetching Binance funding interval for %s",
                symbol,
                exc_info=True,
            )
            return None
        except Exception:
            self.logger().warning(
                "Failed to fetch Binance funding interval for %s",
                symbol,
                exc_info=True,
            )
            return None
        if not isinstance(data, list) or len(data) < 2:
            return None
        times = sorted(
            int(entry["fundingTime"])
            for entry in data
            if isinstance(entry, dict) and "fundingTime" in entry
        )
        if len(times) < 2:
            return None
        interval_ms = abs(times[-1] - times[-2])
        if interval_ms <= 0:
            return None
        interval_hours = Decimal(interval_ms) / Decimal("3600000")
        return interval_hours

    def _maybe_refresh_binance_interval(self, current_time: float):
        connector_key = self.binance_connector_name.lower()
        has_override = connector_key in self._connector_interval_overrides
        if has_override and current_time - self._binance_interval_last_refresh < BINANCE_INTERVAL_REFRESH_SECONDS:
            return
        if current_time - self._binance_interval_last_refresh < BINANCE_INTERVAL_REFRESH_SECONDS:
            return
        try:
            interval_hours = self._fetch_binance_interval_from_api()
        except asyncio.CancelledError:
            raise
        except Exception:
            interval_hours = None
        self._binance_interval_last_refresh = current_time
        if interval_hours is not None:
            self._set_connector_interval_override(self.binance_connector_name, interval_hours)

    def _update_binance_interval_from_next_timestamp(self, next_timestamp: Optional[int], current_time: float):
        if next_timestamp is None:
            return
        if (
            self._binance_last_next_funding_ts is not None
            and next_timestamp > self._binance_last_next_funding_ts
        ):
            interval_seconds = next_timestamp - self._binance_last_next_funding_ts
            if interval_seconds > 0:
                interval_hours = Decimal(interval_seconds) / Decimal(SECONDS_PER_HOUR)
                self._set_connector_interval_override(self.binance_connector_name, interval_hours)
                self._binance_interval_last_refresh = current_time
        self._binance_last_next_funding_ts = next_timestamp

    def _monitor_funding_and_maybe_close(self):
        current_time = self.current_timestamp
        if current_time - self._last_funding_check_ts < 10:
            return
        self._last_funding_check_ts = current_time
        seconds_to_next_hour = self._seconds_to_next_hour(current_time)
        self._latest_funding_eta = seconds_to_next_hour
        self._latest_binance_funding_eta = seconds_to_next_hour
        if seconds_to_next_hour > FUNDING_WINDOW_SECONDS:
            return
        try:
            hyper_info = self.hyperliquid_connector.get_funding_info(self.hyperliquid_trading_pair)
        except Exception:
            hyper_info = None
        if hyper_info is not None:
            self._latest_funding_rate = (
                Decimal(str(hyper_info.rate)) if hyper_info.rate is not None else None
            )
        else:
            self._latest_funding_rate = None

        try:
            binance_info = self.binance_connector.get_funding_info(self.binance_trading_pair)
        except Exception:
            binance_info = None
        if binance_info is not None:
            self._latest_binance_funding_rate = (
                Decimal(str(binance_info.rate)) if binance_info.rate is not None else None
            )
            binance_next_ts = getattr(binance_info, "next_funding_utc_timestamp", None)
            self._update_binance_interval_from_next_timestamp(binance_next_ts, current_time)
        else:
            self._latest_binance_funding_rate = None
        self._maybe_refresh_binance_interval(current_time)
        hyper_hourly_rate = self._hourly_rate_for_connector(
            self._latest_funding_rate,
            self.hyperliquid_connector_name,
        )
        binance_hourly_rate = self._hourly_rate_for_connector(
            self._latest_binance_funding_rate,
            self.binance_connector_name,
        )

        if hyper_hourly_rate is None or binance_hourly_rate is None:
            return

        hyper_hourly_payout = self._funding_payout_value(
            hyper_hourly_rate,
            self.hyperliquid_open_side,
        )
        binance_hourly_payout = self._funding_payout_value(
            binance_hourly_rate,
            self.binance_open_side,
        )

        if hyper_hourly_payout is None or binance_hourly_payout is None:
            return

        net_hourly_payout = hyper_hourly_payout + binance_hourly_payout
        rate_diff = hyper_hourly_rate - binance_hourly_rate

        if net_hourly_payout < Decimal("0"):
            self.logger().info(
                "Hourly funding unfavorable near settlement (rate_diff=%s, net_payout=%s). Closing hedge.",
                self._format_decimal(rate_diff, precision=6, pct=True),
                self._format_decimal(net_hourly_payout, precision=6, pct=True),
            )
            self._initiate_closing("hourly funding unfavorable")

    def _initiate_closing(self, reason: str, resume_open: bool = False) -> bool:
        if self._stage in {"closing", "closed"}:
            return False
        if self._closing_task is not None and not self._closing_task.done():
            return False
        if not self._has_open_exposure():
            self.logger().warning("Cannot close hedge; no filled amount recorded.")
            return False
        self.logger().info("Starting concurrent close: %s", reason)
        self._short_close_fill_event = asyncio.Event()
        self._long_close_fill_event = asyncio.Event()
        self._short_close_order_id = None
        self._long_close_order_id = None
        self._stage = "closing"
        self._closing_recovery = resume_open
        self._closing_task = safe_ensure_future(self._close_positions())
        return True

    async def _close_positions(self):
        closing_success = False
        try:
            await self._ensure_trading_rules_ready()
            hyper_leg = self._leg_for_connector_name(self.hyperliquid_connector_name)
            binance_leg = self._leg_for_connector_name(self.binance_connector_name)
            hyper_recorded = self._short_filled_amount if hyper_leg == "short" else self._long_filled_amount
            binance_recorded = self._short_filled_amount if binance_leg == "short" else self._long_filled_amount

            hyper_close_amount = await self._determine_close_amount(
                connector=self.hyperliquid_connector,
                trading_pair=self.hyperliquid_trading_pair,
                recorded_amount=hyper_recorded,
            )
            binance_close_amount = await self._determine_close_amount(
                connector=self.binance_connector,
                trading_pair=self.binance_trading_pair,
                recorded_amount=binance_recorded,
            )

            connector_close_amounts = {
                self.hyperliquid_connector_name: hyper_close_amount,
                self.binance_connector_name: binance_close_amount,
            }
            short_close_amount = connector_close_amounts[self.short_connector_name]
            long_close_amount = connector_close_amounts[self.long_connector_name]

            if short_close_amount <= Decimal("0") and long_close_amount <= Decimal("0"):
                self.logger().info("No positions detected; marking hedge as closed.")
                if not self._closing_recovery:
                    self._stage = "closed"
                    self._execution_completed = True
                closing_success = True
                return

            close_results = await asyncio.gather(
                self._execute_limit_order(
                    connector_name=self.hyperliquid_connector_name,
                    trading_pair=self.hyperliquid_trading_pair,
                    side=self.hyperliquid_close_side,
                    amount=connector_close_amounts[self.hyperliquid_connector_name],
                    position_action=PositionAction.CLOSE,
                    leg=self._leg_for_connector_name(self.hyperliquid_connector_name),
                    stage="close",
                ) if connector_close_amounts[self.hyperliquid_connector_name] > Decimal("0") else asyncio.sleep(0, result=True),
                self._execute_limit_order(
                    connector_name=self.binance_connector_name,
                    trading_pair=self.binance_trading_pair,
                    side=self.binance_close_side,
                    amount=connector_close_amounts[self.binance_connector_name],
                    position_action=PositionAction.CLOSE,
                    leg=self._leg_for_connector_name(self.binance_connector_name),
                    stage="close",
                ) if connector_close_amounts[self.binance_connector_name] > Decimal("0") else asyncio.sleep(0, result=True),
            )
            closing_success = all(close_results)
            if closing_success:
                self.logger().info("Hedge closed successfully.")
            else:
                self.logger().warning("Closing tasks did not fully succeed (results=%s).", close_results)
            if not self._closing_recovery:
                self._stage = "closed"
        except Exception as exc:
            self.logger().error("Exception while closing positions: %s", exc, exc_info=True)
            self._stage = "closing_failed"
        finally:
            if self._closing_recovery:
                self._closing_recovery = False
                if closing_success:
                    self._reset_cycle_state()
                    self._stage = "not_started"
                else:
                    self._stage = "closing_failed"
                self._execution_completed = False
            else:
                self._execution_completed = True

    async def _determine_close_amount(
        self,
        connector: ConnectorBase,
        trading_pair: str,
        recorded_amount: Optional[Decimal],
    ) -> Decimal:
        position_amount: Optional[Decimal] = None
        try:
            position = getattr(connector, "get_position", lambda _: None)(trading_pair)
            if position is not None and getattr(position, "amount", None) is not None:
                position_amount = Decimal(str(position.amount)).copy_abs()
        except Exception:
            position_amount = None
        if position_amount is None or position_amount <= Decimal("0"):
            position_amount = recorded_amount or Decimal("0")
        rule = self._short_trading_rule if connector is self.short_connector else self._long_trading_rule
        if position_amount is None or position_amount <= Decimal("0"):
            return Decimal("0")
        amount = self._adjust_amount_for_rule(position_amount, None, rule, round_up=True, ensure_min_notional=False)
        return amount or Decimal("0")

    def did_complete_funding_payment(self, event: FundingPaymentCompletedEvent):
        if event.trading_pair not in {self.hyperliquid_trading_pair, self.binance_trading_pair}:
            return
        amount = Decimal(str(event.amount))
        self._funding_total_quote += amount
        self.logger().info(
            "Funding payment %s %s on %s. Cumulative funding: %s",
            self._format_decimal(amount),
            event.trading_pair.split("-")[-1],
            event.market,
            self._format_decimal(self._funding_total_quote),
        )

    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        self._strategy_cancelled_orders.discard(event.order_id)
        owner = self._order_owner.pop(event.order_id, None)
        if not owner:
            return
        leg, stage = owner
        base_amount_raw = getattr(event, "base_asset_amount", getattr(event, "amount", None))
        quote_amount_raw = getattr(event, "quote_asset_amount", None)
        if base_amount_raw is None:
            return
        base_amount = Decimal(str(base_amount_raw))
        if stage == "open" and leg == "short":
            self._short_filled_amount = base_amount
            if quote_amount_raw is not None and base_amount > Decimal("0"):
                quote_amount = Decimal(str(quote_amount_raw))
                self._short_entry_price = quote_amount / base_amount
                self._short_target_notional = self._short_entry_price * base_amount
            if not self._short_fill_event.is_set():
                self._short_fill_event.set()
        elif stage == "close" and leg == "long":
            self._long_close_filled_amount = (self._long_close_filled_amount or Decimal("0")) + base_amount
            if quote_amount_raw is not None and base_amount > Decimal("0"):
                quote_amount = Decimal(str(quote_amount_raw))
                self._long_close_order_price = quote_amount / base_amount
            if not self._long_close_fill_event.is_set():
                self._long_close_fill_event.set()

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        self._strategy_cancelled_orders.discard(event.order_id)
        owner = self._order_owner.pop(event.order_id, None)
        if not owner:
            return
        leg, stage = owner
        base_amount_raw = getattr(event, "base_asset_amount", getattr(event, "amount", None))
        quote_amount_raw = getattr(event, "quote_asset_amount", None)
        if base_amount_raw is None:
            return
        base_amount = Decimal(str(base_amount_raw))
        if stage == "open" and leg == "long":
            self._long_filled_amount = base_amount
            if quote_amount_raw is not None and base_amount > Decimal("0"):
                quote_amount = Decimal(str(quote_amount_raw))
                self._long_entry_price = quote_amount / base_amount
                self._long_target_notional = self._long_entry_price * base_amount
            if not self._long_fill_event.is_set():
                self._long_fill_event.set()
        elif stage == "close" and leg == "short":
            self._short_close_filled_amount = (self._short_close_filled_amount or Decimal("0")) + base_amount
            if quote_amount_raw is not None and base_amount > Decimal("0"):
                quote_amount = Decimal(str(quote_amount_raw))
                self._short_close_order_price = quote_amount / base_amount
            if not self._short_close_fill_event.is_set():
                self._short_close_fill_event.set()

    def did_cancel_order(self, event: OrderCancelledEvent):
        expected = event.order_id in self._strategy_cancelled_orders
        if expected:
            self._strategy_cancelled_orders.discard(event.order_id)
        owner = self._order_owner.pop(event.order_id, None)
        if not owner:
            return
        leg, stage = owner
        order_id_attr, _, fill_event = self._leg_attributes(leg, stage)
        setattr(self, order_id_attr, None)
        if not expected and not fill_event.is_set():
            fill_event.set()

    def did_fail_order(self, event: MarketOrderFailureEvent):
        self._strategy_cancelled_orders.discard(event.order_id)
        owner = self._order_owner.pop(event.order_id, None)
        if not owner:
            return
        leg, stage = owner
        error_message = getattr(event, "error_message", "") or ""
        if stage == "open" and self._is_post_only_rejection(error_message):
            self.logger().debug(
                "Order %s (%s, %s) rejected as taker risk (post-only). Will reprice.",
                event.order_id,
                leg,
                stage,
            )
            return
        self.logger().warning("Order %s (%s, %s) failed. Resetting state.", event.order_id, leg, stage)
        if stage == "open":
            self._stage = "not_started"
        else:
            self._stage = "closing_failed"

    def format_status(self) -> str:
        original_status = super().format_status()
        lines = []
        stage_labels = {
            "not_started": "未开始",
            "opening": "建仓中",
            "hedged": "已对冲",
            "closing": "平仓中",
            "closed": "已完成",
            "closing_failed": "平仓失败",
        }
        lines.append(f"当前阶段: {stage_labels.get(self._stage, self._stage)}")
        if self._stage in {"opening", "hedged", "closing"}:
            hyper_interval = self._settlement_hours_for_connector(self.hyperliquid_connector_name)
            binance_interval = self._settlement_hours_for_connector(self.binance_connector_name)
            hyper_hourly_rate = self._hourly_rate_for_connector(
                self._latest_funding_rate,
                self.hyperliquid_connector_name,
            )
            binance_hourly_rate = self._hourly_rate_for_connector(
                self._latest_binance_funding_rate,
                self.binance_connector_name,
            )
            lines.append(
                f"空腿仓位: 数量={self._format_decimal(self._short_filled_amount or Decimal('0'))} @ {self._format_decimal(self._short_entry_price) if self._short_entry_price else '暂缺'}"
            )
            lines.append(
                f"多腿仓位: 数量={self._format_decimal(self._long_filled_amount or Decimal('0'))} @ {self._format_decimal(self._long_entry_price) if self._long_entry_price else '暂缺'}"
            )
            if self._latest_funding_rate is not None:
                lines.append(
                    f"短腿最新资金费率: {self._format_decimal(self._latest_funding_rate, precision=6, pct=True)}"
                )
            if hyper_hourly_rate is not None:
                lines.append(
                    f"短腿小时化资金费率: {self._format_decimal(hyper_hourly_rate, precision=6, pct=True)}"
                )
            if hyper_interval is not None:
                lines.append(f"短腿结算周期: {self._format_decimal(hyper_interval)} 小时")
            if self._latest_funding_eta is not None:
                lines.append(f"距离下次短腿结算(秒): {self._latest_funding_eta}")
            if self._latest_binance_funding_rate is not None:
                lines.append(
                    f"长腿最新资金费率: {self._format_decimal(self._latest_binance_funding_rate, precision=6, pct=True)}"
                )
            if binance_hourly_rate is not None:
                lines.append(
                    f"长腿小时化资金费率: {self._format_decimal(binance_hourly_rate, precision=6, pct=True)}"
                )
            if binance_interval is not None:
                lines.append(f"长腿结算周期: {self._format_decimal(binance_interval)} 小时")
            if self._latest_binance_funding_eta is not None:
                lines.append(f"距离下次长腿结算(秒): {self._latest_binance_funding_eta}")
            lines.append(f"累计资金费盈亏: {self._format_decimal(self._funding_total_quote)}")
        return original_status + "\n".join(lines)

    def close_positions(self):
        self._initiate_closing("user requested close")

    def _calculate_amount_from_margin(
        self,
        connector_name: str,
        trading_pair: str,
        side: TradeType,
        target_margin: Decimal,
        price: Decimal,
        leverage: Decimal,
    ) -> Optional[Decimal]:
        leg = self._leg_for_connector_name(connector_name)
        rule = self._short_trading_rule if leg == "short" else self._long_trading_rule
        if rule is None:
            return None
        if price is None or price <= Decimal("0"):
            return None
        notional_target = target_margin * leverage
        amount = notional_target / price
        amount = self._adjust_amount_for_rule(amount, price, rule, round_up=True, ensure_min_notional=True)
        return amount

    def _get_market_price(self, connector_name: str, trading_pair: str, side: TradeType) -> Optional[Decimal]:
        price_type = PriceType.BestAsk if side == TradeType.BUY else PriceType.BestBid
        value = self.market_data_provider.get_price_by_type(connector_name=connector_name, trading_pair=trading_pair, price_type=price_type)
        is_short_connector = connector_name == self.short_connector_name
        if value is None:
            return self._last_short_price if is_short_connector else self._last_long_price
        price = Decimal(str(value))
        if price > Decimal("0"):
            if is_short_connector:
                self._last_short_price = price
            else:
                self._last_long_price = price
            return price
        return self._last_short_price if is_short_connector else self._last_long_price

    def _apply_price_rules(self, price: Decimal, rule, side: TradeType) -> Optional[Decimal]:
        if price is None or price <= Decimal("0"):
            return None
        step = self._decimal_from_rule_value(getattr(rule, "min_price_increment", None))
        if step and step > Decimal("0"):
            rounding = ROUND_FLOOR if side == TradeType.BUY else ROUND_CEILING
            price = (price / step).to_integral_value(rounding=rounding) * step
            if side == TradeType.BUY:
                candidate = price - step
                if candidate > Decimal("0"):
                    price = candidate
            else:
                price = price + step
        return price if price > Decimal("0") else None

    def _estimate_price_from_cache(self, leg: str) -> Optional[Decimal]:
        price = self._last_short_price if leg == "short" else self._last_long_price
        rule = self._short_trading_rule if leg == "short" else self._long_trading_rule
        side = self._short_open_side if leg == "short" else self._long_open_side
        if price is None or price <= Decimal("0"):
            return None
        if rule is None:
            return price
        return self._apply_price_rules(price, rule, side)

    def _counterpart_reference_price(self, leg: str) -> Optional[Decimal]:
        if leg == "short":
            sources = [
                self._long_entry_price,
                self._long_open_order_price,
                self._estimate_price_from_cache("long"),
            ]
        else:
            sources = [
                self._short_entry_price,
                self._short_open_order_price,
                self._estimate_price_from_cache("short"),
            ]
        for value in sources:
            if value is not None and value > Decimal("0"):
                return value
        return None

    def _meets_entry_spread_requirement(self, leg: str, stage: str, candidate_price: Decimal) -> bool:
        if stage != "open" or self._min_entry_spread <= Decimal("0"):
            return True
        counterpart_price = self._counterpart_reference_price(leg)
        if counterpart_price is None or counterpart_price <= Decimal("0"):
            return False
        ratio = Decimal("1") + self._min_entry_spread
        if leg == "short":
            return candidate_price >= counterpart_price * ratio
        return candidate_price <= counterpart_price / ratio

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

    @staticmethod
    def _decimal_from_rule_value(value: Optional[Decimal]) -> Optional[Decimal]:
        if value is None:
            return None
        try:
            decimal_value = Decimal(str(value))
        except Exception:
            return None
        return decimal_value if decimal_value > Decimal("0") else None

    @staticmethod
    def _is_post_only_rejection(error_message: Optional[str]) -> bool:
        if not error_message:
            return False
        lowered = error_message.lower()
        return "post only" in lowered or "-5022" in lowered

    def _format_decimal(self, value: Optional[Decimal], precision: int = 6, pct: bool = False) -> str:
        if value is None:
            return "n/a"
        quant = Decimal("1") / (Decimal("10") ** precision)
        if pct:
            return f"{(value * Decimal('100')).quantize(quant, rounding=ROUND_HALF_UP)}%"
        return str(value.quantize(quant, rounding=ROUND_HALF_UP))
