"""
tiny-statemachine — Zero-dependency state machine for Python AI agent behavior trees.
Define states, transitions, guards, and actions — in one file.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

__version__ = "1.0.0"
__all__ = [
    "StateMachine",
    "state",
    "transition",
    "substate",
    "Event",
    "InvalidTransition",
    "GuardFailed",
    "StateNotFound",
]


# ─── Exceptions ──────────────────────────────────────────────────────────────


class StateMachineError(Exception):
    """Base exception for state machine errors."""
    pass


class InvalidTransition(StateMachineError):
    """Raised when no transition exists for the current state + event."""

    def __init__(self, state: str, event: str):
        self.state = state
        self.event = event
        super().__init__(
            f"No transition from state '{state}' on event '{event}'"
        )


class GuardFailed(StateMachineError):
    """Raised when a guard condition returns False."""

    def __init__(self, guard_name: str, state: str, event: str):
        self.guard_name = guard_name
        super().__init__(
            f"Guard '{guard_name}' failed: transition from '{state}' on '{event}' blocked"
        )


class StateNotFound(StateMachineError):
    """Raised when a referenced state doesn't exist."""

    def __init__(self, state_name: str):
        super().__init__(f"State '{state_name}' not found")


# ─── Event ───────────────────────────────────────────────────────────────────


@dataclass
class Event:
    """An event that can trigger transitions."""

    name: str
    data: Any = None
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        if self.data is not None:
            return f"Event({self.name!r}, data={self.data!r})"
        return f"Event({self.name!r})"


# ─── Transition Definition ───────────────────────────────────────────────────


@dataclass
class TransitionDef:
    """Definition of a transition."""

    from_state: str
    to_state: str
    event_name: str
    guard: Optional[str] = None
    unless: Optional[str] = None
    action: Optional[Callable] = None


# ─── State Definition ────────────────────────────────────────────────────────


@dataclass
class StateDef:
    """Definition of a state."""

    name: str
    on_enter: Optional[str] = None
    on_exit: Optional[str] = None
    parent: Optional[str] = None
    substate_names: List[str] = field(default_factory=list)


# ─── Decorators ──────────────────────────────────────────────────────────────


def state(on_enter: Optional[str] = None, on_exit: Optional[str] = None):
    """Decorator to mark a method as a state."""

    def decorator(fn: Callable) -> Callable:
        fn._is_state = True
        fn._on_enter = on_enter
        fn._on_exit = on_exit
        fn._state_name = fn.__name__
        return fn

    return decorator


def substate(fn: Callable) -> Callable:
    """Decorator to mark a method as a substate of the parent composite state."""
    fn._is_substate = True
    return fn


def transition(
    from_state: str,
    to_state: str,
    event: str,
    guard: Optional[str] = None,
    unless: Optional[str] = None,
):
    """Decorator to mark a method as a transition action and register the transition."""

    def decorator(fn: Callable) -> Callable:
        fn._is_transition = True
        fn._from_state = from_state
        fn._to_state = to_state
        fn._event_name = event
        fn._guard = guard
        fn._unless = unless
        return fn

    return decorator


# ─── State Machine ────────────────────────────────────────────────────────────


class StateMachine:
    """
    A state machine with hierarchical states, guards, and entry/exit actions.

    Subclass and define states with @state, transitions with @transition.
    """

    initial: str = "idle"

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        self._states: Dict[str, StateDef] = {}
        self._transitions: Dict[Tuple[str, str], List[TransitionDef]] = {}
        self._transition_actions: Dict[str, Callable] = {}
        self._current: str = getattr(self, "initial", "idle")
        self._current_substate: Optional[str] = None
        self._context: Dict[str, Any] = context or {}
        self._event: Optional[Event] = None
        self._history: List[Tuple[str, str, str]] = []
        self._substate_parents: Dict[str, str] = {}
        self._composite_states: Set[str] = set()

        self._discover()

    # ─── Discovery ─────────────────────────────────────────────────────────

    def _discover(self) -> None:
        """Scan the class for @state and @transition decorators."""
        # Find composite states (those with substate methods)
        for name in dir(self):
            if name.startswith("_"):
                continue
            attr = getattr(self, name, None)
            if callable(attr) and hasattr(attr, "_is_state"):
                # This is a composite state if any method has _is_substate
                substate_names = []
                for sub_name in dir(self):
                    sub_attr = getattr(self, sub_name, None)
                    if callable(sub_attr) and hasattr(sub_attr, "_is_substate"):
                        parent = getattr(sub_attr, "__self__", None)
                        if parent and parent.__class__ == self.__class__:
                            # Check if it's under this state via naming convention
                            if name in sub_attr.__name__:
                                substate_names.append(sub_attr.__name__)
                if substate_names:
                    self._composite_states.add(name)

        for name in dir(self):
            if name.startswith("_"):
                continue
            attr = getattr(self, name, None)
            if not callable(attr):
                continue

            if hasattr(attr, "_is_state") and attr._is_state:
                sname = attr._state_name
                self._states[sname] = StateDef(
                    name=sname,
                    on_enter=attr._on_enter,
                    on_exit=attr._on_exit,
                )

            if hasattr(attr, "_is_transition") and attr._is_transition:
                tdef = TransitionDef(
                    from_state=attr._from_state,
                    to_state=attr._to_state,
                    event_name=attr._event_name,
                    guard=attr._guard,
                    unless=attr._unless,
                    action=attr,
                )
                key = (attr._from_state, attr._event_name)
                if key not in self._transitions:
                    self._transitions[key] = []
                self._transitions[key].append(tdef)
                self._transition_actions[attr.__name__] = attr

    # ─── Properties ─────────────────────────────────────────────────────────

    @property
    def current(self) -> str:
        """Current state name."""
        if self._current_substate:
            return self._current_substate
        return self._current

    @current.setter
    def current(self, value: str) -> None:
        if value not in self._states and value not in self._substate_parents:
            raise StateNotFound(value)
        # If it's a substate, also set parent
        if value in self._substate_parents:
            self._current = self._substate_parents[value]
            self._current_substate = value
        else:
            self._current = value
            self._current_substate = None

    @property
    def current_substate(self) -> Optional[str]:
        """Current substate (if in a composite state)."""
        return self._current_substate

    @current_substate.setter
    def current_substate(self, value: Optional[str]) -> None:
        self._current_substate = value

    @property
    def context(self) -> Dict[str, Any]:
        """Arbitrary key/value storage for the machine."""
        return self._context

    @property
    def event(self) -> Optional[Event]:
        """Last triggered event."""
        return self._event

    @event.setter
    def event(self, value: Any) -> None:
        """Set event from arbitrary value (auto-wraps dicts)."""
        if isinstance(value, Event):
            self._event = value
        else:
            self._event = Event("external", data=value)

    @property
    def history(self) -> List[Tuple[str, str, str]]:
        """List of (from_state, event_name, to_state) for past transitions."""
        return self._history.copy()

    # ─── Transition Logic ──────────────────────────────────────────────────

    def _run_callback(self, callback_name: Optional[str]) -> None:
        """Run a named callback method."""
        if callback_name and hasattr(self, callback_name):
            cb = getattr(self, callback_name)
            if callable(cb):
                result = cb()
                # Allow callback to return False to veto the transition
                if result is False:
                    raise GuardFailed(callback_name, self._current, getattr(self._event, 'name', '?'))

    def _check_guard(self, guard_name: Optional[str]) -> bool:
        """Evaluate a guard condition."""
        if guard_name is None:
            return True
        if hasattr(self, guard_name):
            guard_fn = getattr(self, guard_name)
            if callable(guard_fn):
                return bool(guard_fn())
        return True

    def _run_transition(self, tdef: TransitionDef) -> None:
        """Execute a single transition."""
        # Fire exit action of old state
        old_state_def = self._states.get(self._current)
        if old_state_def:
            self._run_callback(old_state_def.on_exit)

        from_state = self._current
        to_state = tdef.to_state

        # Execute transition action
        if tdef.action:
            tdef.action(self)

        # Update state
        self._current = to_state
        self._current_substate = None

        # Fire enter action of new state
        new_state_def = self._states.get(to_state)
        if new_state_def:
            self._run_callback(new_state_def.on_enter)

        # Record history
        event_name = getattr(self._event, "name", tdef.event_name)
        self._history.append((from_state, event_name, to_state))

    # ─── Public API ─────────────────────────────────────────────────────────

    def trigger(self, event_name: str, **data) -> None:
        """
        Fire an event. Triggers the first matching transition from current state.
        Raises InvalidTransition if no transition exists.
        Raises GuardFailed if guard condition returns False.
        """
        self._event = Event(event_name, data=data if data else None)

        # Check if current state is a composite with active substate
        active_state = self._current_substate or self._current
        key = (active_state, event_name)

        # First try exact state match
        if key in self._transitions:
            for tdef in self._transitions[key]:
                # Check guard
                if not self._check_guard(tdef.guard):
                    continue
                # Check unless
                if tdef.unless and self._check_guard(tdef.unless):
                    continue
                self._run_transition(tdef)
                return

        # Try parent state (for substate transitions)
        if active_state in self._substate_parents:
            parent = self._substate_parents[active_state]
            key = (parent, event_name)
            if key in self._transitions:
                for tdef in self._transitions[key]:
                    if not self._check_guard(tdef.guard):
                        continue
                    if tdef.unless and self._check_guard(tdef.unless):
                        continue
                    self._run_transition(tdef)
                    return

        # No matching transition
        raise InvalidTransition(self._current, event_name)

    def can(self, event_name: str) -> bool:
        """
        Returns True if the event would trigger a valid transition from current state.
        Does not execute the transition.
        """
        active_state = self._current_substate or self._current
        key = (active_state, event_name)
        if key in self._transitions:
            for tdef in self._transitions[key]:
                if self._check_guard(tdef.guard):
                    return True
        return False

    def active_states(self) -> List[str]:
        """Returns list of currently active states (including parent)."""
        states = [self._current]
        if self._current_substate:
            states.append(self._current_substate)
        return states

    def state_of(self, name: str) -> Optional[StateDef]:
        """Get state definition by name."""
        return self._states.get(name)

    def is_in(self, *state_names: str) -> bool:
        """Returns True if current state matches any of the given names."""
        active = self._current_substate or self._current
        return active in state_names

    def reset(self) -> None:
        """Reset to initial state."""
        self._current = getattr(self, "initial", "idle")
        self._current_substate = None
        self._event = None
        self._history.clear()

    def __repr__(self) -> str:
        sub = f"/{self._current_substate}" if self._current_substate else ""
        return f"<{self.__class__.__name__}: {self._current}{sub}>"
