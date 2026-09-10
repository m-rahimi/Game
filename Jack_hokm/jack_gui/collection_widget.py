from __future__ import annotations

from kivy.uix.floatlayout import FloatLayout

from .card_widget import CardWidget
from .layout import trick_collect_layout, viewport_size, TRICK_COLLECT_DURATION


class CollectionWidget(FloatLayout):
    """Won tricks stacked at bottom-left (team 0) and top-left (team 1)."""

    def __init__(self, **kwargs):
        super().__init__(size_hint=(1, 1), **kwargs)
        self.team_trick_count: dict[int, int] = {0: 0, 1: 0}
        self.collected: list[tuple[CardWidget, int, int]] = []

    def relayout(self, *_args):
        if viewport_size()[0] <= 0 or viewport_size()[1] <= 0:
            return
        for widget, team, trick_index in self.collected:
            size, pos, rotation = trick_collect_layout(team, trick_index)
            widget.snap_layout(pos, size=size, rotation=rotation)

    def clear(self):
        for widget, *_meta in self.collected:
            self.remove_widget(widget)
        self.collected = []
        self.team_trick_count = {0: 0, 1: 0}

    def absorb_trick(
        self,
        widgets: list[CardWidget],
        winning_team: int,
        callback,
    ):
        if not widgets:
            callback()
            return

        trick_index = self.team_trick_count[winning_team]
        self.team_trick_count[winning_team] += 1

        pending = len(widgets)
        finished = [0]

        def one_done():
            finished[0] += 1
            if finished[0] >= pending:
                callback()

        for widget in widgets:
            widget.set_card_back()
            size, pos, rotation = trick_collect_layout(winning_team, trick_index)
            widget.name = "Collected"
            widget.playable = False
            widget.rotation = rotation
            widget.size = size
            if widget.parent:
                widget.parent.remove_widget(widget)
            self.add_widget(widget)
            self.collected.append((widget, winning_team, trick_index))
            widget.animate_to(pos, rotation=rotation, duration=TRICK_COLLECT_DURATION, callback=one_done)
