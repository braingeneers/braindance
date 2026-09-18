# racing_game.py

**Path:** `braindance/games/racing_game.py`
**Module:** `braindance.games.racing_game`
**Feature Area:** `Games and Reinforcement`
**Entry point:** yes — main / module execution (static candidate)

## Overview
Implements a standalone Pygame racing game with procedural oval track, checkpoints, obstacles, collectibles and boost. Updates car physics, a following camera, minimap and scoring HUD from keyboard input.

## Connections
- **Shared data:** Game.run -> handle_events -> Car.update/Track collision checks -> camera -> rendering.

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.
- `pygame` — external or unresolved local import; source import evidence.

## Classes
### GameState(Enum)
> unclear — see source
**Source:** `braindance/games/racing_game.py:35`
**Kind:** class. **Instantiated by:** unclear — see source
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
None
### Point()
> unclear — see source
**Source:** `braindance/games/racing_game.py:41`
**Kind:** class. **Instantiated by:** braindance/games/racing_game.py:1102 (named-call hint); braindance/games/racing_game.py:1107 (named-call hint); braindance/games/racing_game.py:1112 (named-call hint); braindance/games/racing_game.py:1116 (named-call hint); braindance/games/racing_game.py:1119 (named-call hint); braindance/games/racing_game.py:1145 (named-call hint); braindance/games/racing_game.py:1161 (named-call hint); braindance/games/racing_game.py:179 (named-call hint); braindance/games/racing_game.py:180 (named-call hint); braindance/games/racing_game.py:181 (named-call hint); braindance/games/racing_game.py:182 (named-call hint); braindance/games/racing_game.py:183 (named-call hint); braindance/games/racing_game.py:189 (named-call hint); braindance/games/racing_game.py:242 (named-call hint); braindance/games/racing_game.py:263 (named-call hint); braindance/games/racing_game.py:376 (named-call hint); braindance/games/racing_game.py:410 (named-call hint); braindance/games/racing_game.py:415 (named-call hint); braindance/games/racing_game.py:420 (named-call hint); braindance/games/racing_game.py:424 (named-call hint); braindance/games/racing_game.py:430 (named-call hint); braindance/games/racing_game.py:461 (named-call hint); braindance/games/racing_game.py:482 (named-call hint); braindance/games/racing_game.py:503 (named-call hint); braindance/games/racing_game.py:52 (named-call hint); braindance/games/racing_game.py:523 (named-call hint); braindance/games/racing_game.py:55 (named-call hint); braindance/games/racing_game.py:551 (named-call hint); braindance/games/racing_game.py:555 (named-call hint); braindance/games/racing_game.py:696 (named-call hint); braindance/games/racing_game.py:727 (named-call hint); braindance/games/racing_game.py:799 (named-call hint)
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
#### `distance_to(self, other: 'Point') -> float`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:45`
#### `as_tuple(self) -> Tuple[float, float]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:48`
#### `add(self, other: 'Point') -> 'Point'`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:51`
#### `scale(self, factor: float) -> 'Point'`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:54`
### TrackSegment()
> unclear — see source
**Source:** `braindance/games/racing_game.py:57`
**Kind:** class. **Instantiated by:** braindance/games/racing_game.py:1070 (named-call hint); braindance/games/racing_game.py:427 (named-call hint)
**Constructor:** `__init__(self, outer_points: List[Point], inner_points: List[Point], center_points: List[Point])`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `center_points` | inferred at runtime | `center_points` |
| `inner_points` | inferred at runtime | `inner_points` |
| `outer_points` | inferred at runtime | `outer_points` |
**Methods:**
#### `draw(self, surface: pygame.Surface, camera_offset: Point)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:63`
### Obstacle()
> unclear — see source
**Source:** `braindance/games/racing_game.py:78`
**Kind:** class. **Instantiated by:** braindance/games/racing_game.py:1145 (named-call hint); braindance/games/racing_game.py:460 (named-call hint); braindance/games/racing_game.py:481 (named-call hint)
**Constructor:** `__init__(self, position: Point, size: float, obstacle_type: str='rock')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `color` | inferred at runtime | `BROWN if obstacle_type == 'rock' else RED` |
| `hitbox_radius` | inferred at runtime | `size / 2` |
| `position` | inferred at runtime | `position` |
| `size` | inferred at runtime | `size` |
| `type` | inferred at runtime | `obstacle_type` |
**Methods:**
#### `draw(self, surface: pygame.Surface, camera_offset: Point)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:86`
#### `check_collision(self, car_position: Point, car_radius: float) -> bool`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:106`
### Collectible()
> unclear — see source
**Source:** `braindance/games/racing_game.py:109`
**Kind:** class. **Instantiated by:** braindance/games/racing_game.py:1160 (named-call hint); braindance/games/racing_game.py:502 (named-call hint); braindance/games/racing_game.py:522 (named-call hint)
**Constructor:** `__init__(self, position: Point, collectible_type: str='coin')`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `active` | inferred at runtime | `True` |
| `animation_time` | inferred at runtime | `0` |
| `color` | inferred at runtime | `YELLOW if collectible_type == 'coin' else ORANGE` |
| `position` | inferred at runtime | `position` |
| `radius` | inferred at runtime | `10` |
| `type` | inferred at runtime | `collectible_type` |
**Methods:**
#### `draw(self, surface: pygame.Surface, camera_offset: Point, dt: float)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/racing_game.py:118`
#### `check_collision(self, car_position: Point, car_radius: float) -> bool`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:144`
### Car()
> unclear — see source
**Source:** `braindance/games/racing_game.py:149`
**Kind:** class. **Instantiated by:** braindance/games/racing_game.py:798 (named-call hint)
**Constructor:** `__init__(self, position: Point, angle: float=0.0)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `acceleration` | inferred at runtime | `0.0` |
| `angle` | inferred at runtime | `angle` |
| `boost` | inferred at runtime | `0.0` |
| `boosting` | inferred at runtime | `False` |
| `car_points` | inferred at runtime | `self._compute_car_points()` |
| `collision_radius` | inferred at runtime | `20` |
| `drift_factor` | inferred at runtime | `0.95` |
| `length` | inferred at runtime | `40` |
| `max_reverse_velocity` | inferred at runtime | `-5.0` |
| `max_velocity` | inferred at runtime | `10.0` |
| `position` | inferred at runtime | `position` |
| `steering` | inferred at runtime | `0.0` |
| `trail_max_points` | inferred at runtime | `50` |
| `trail_points` | inferred at runtime | `[]` |
| `trail_timer` | inferred at runtime | `0` |
| `velocity` | inferred at runtime | `0.0` |
| `width` | inferred at runtime | `20` |
**Methods:**
#### `_compute_car_points(self) -> List[Point]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:173`
#### `update(self, gas: float, steering: float, dt: float, track: 'Track')`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/racing_game.py:187`
#### `draw(self, surface: pygame.Surface, camera_offset: Point=None)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:258`
#### `get_distance_to_center(self, track: 'Track') -> float`
> Get distance from car to closest point on track center line
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:326`
#### `activate_boost(self)`
> Activate boost if available
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/racing_game.py:330`
### BackgroundGrid()
> unclear — see source
**Source:** `braindance/games/racing_game.py:336`
**Kind:** class. **Instantiated by:** braindance/games/racing_game.py:792 (named-call hint)
**Constructor:** `__init__(self, cell_size=100)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `cell_size` | inferred at runtime | `cell_size` |
**Methods:**
#### `draw(self, surface: pygame.Surface, camera_offset: Point)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:340`
### Track()
> unclear — see source
**Source:** `braindance/games/racing_game.py:364`
**Kind:** class. **Instantiated by:** braindance/games/racing_game.py:795 (named-call hint)
**Constructor:** `__init__(self, width: int, height: int)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `center_points` | inferred at runtime | `[]` |
| `checkpoints` | inferred at runtime | `[]` |
| `collectibles` | inferred at runtime | `[]` |
| `height` | inferred at runtime | `height` |
| `inner_points` | inferred at runtime | `[]` |
| `obstacles` | inferred at runtime | `[]` |
| `outer_points` | inferred at runtime | `[]` |
| `segments` | inferred at runtime | `[]` |
| `start_angle` | inferred at runtime | `0` |
| `start_finish_line` | inferred at runtime | `[Point(start_point.x - perp_dx * line_width / 2, start_point.y - perp_dy * line_width / 2), Point(start_point.x + perp_dx * line_…` |
| `start_position` | inferred at runtime | `Point(0, 0)` |
| `track_width` | inferred at runtime | `120` |
| `width` | inferred at runtime | `width` |
**Methods:**
#### `_generate_track(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/racing_game.py:382`
#### `_add_obstacles(self)`
> Add obstacles around the track
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:446`
#### `_add_collectibles(self)`
> Add collectibles on the track
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:488`
#### `_add_start_finish_line(self)`
> Add a start/finish line to the track
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/racing_game.py:528`
#### `draw(self, surface: pygame.Surface, camera_offset: Point, dt: float=0)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:561`
#### `is_point_on_track(self, point: Point) -> bool`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:612`
#### `_point_in_polygon(self, point: Point, polygon: List[Point]) -> bool`
> Check if point is inside a polygon using ray casting algorithm Credit: Modified from https://en.wikipedia.org/wiki/Point_in_polygon
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:619`
#### `distance_to_center(self, point: Point) -> float`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:640`
#### `check_lap(self, car_position: Point, last_checkpoint_idx: int) -> Tuple[bool, int]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:653`
#### `check_obstacle_collisions(self, car: Car) -> Optional[Obstacle]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:673`
#### `check_collectible_collisions(self, car: Car) -> List[Collectible]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:679`
#### `reset_collectibles(self)`
> Reset all collectibles to active state
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:687`
### Camera()
> unclear — see source
**Source:** `braindance/games/racing_game.py:692`
**Kind:** class. **Instantiated by:** braindance/games/racing_game.py:804 (named-call hint)
**Constructor:** `__init__(self, screen_width: int, screen_height: int)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `height` | inferred at runtime | `screen_height` |
| `position` | inferred at runtime | `Point(0, 0)` |
| `smoothness` | inferred at runtime | `0.05` |
| `target` | inferred at runtime | `None` |
| `width` | inferred at runtime | `screen_width` |
**Methods:**
#### `follow(self, target: Point)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/racing_game.py:700`
#### `update(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/racing_game.py:703`
#### `get_offset(self) -> Point`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:718`
### Minimap()
> unclear — see source
**Source:** `braindance/games/racing_game.py:721`
**Kind:** class. **Instantiated by:** braindance/games/racing_game.py:808 (named-call hint)
**Constructor:** `__init__(self, width: int, height: int, scale: float=0.15)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `height` | inferred at runtime | `height` |
| `position` | inferred at runtime | `Point(SCREEN_WIDTH - width - 10, 10)` |
| `scale` | inferred at runtime | `scale` |
| `surface` | inferred at runtime | `pygame.Surface((width, height), pygame.SRCALPHA)` |
| `width` | inferred at runtime | `width` |
**Methods:**
#### `draw(self, screen: pygame.Surface, track: Track, car: Car)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:729`
### Game()
> unclear — see source
**Source:** `braindance/games/racing_game.py:783`
**Kind:** class. **Instantiated by:** braindance/games/racing_game.py:1375 (named-call hint)
**Constructor:** `__init__(self)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `bg_grid` | inferred at runtime | `BackgroundGrid()` |
| `boost_sound` | inferred at runtime | `pygame.mixer.Sound('boost.wav')` |
| `camera` | inferred at runtime | `Camera(SCREEN_WIDTH, SCREEN_HEIGHT)` |
| `clock` | inferred at runtime | `pygame.time.Clock()` |
| `coin_sound` | inferred at runtime | `pygame.mixer.Sound('coin.wav')` |
| `collision_cooldown` | inferred at runtime | `0` |
| `crash_sound` | inferred at runtime | `pygame.mixer.Sound('crash.wav')` |
| `font` | inferred at runtime | `pygame.font.Font(None, 36)` |
| `lap_count` | inferred at runtime | `0` |
| `last_checkpoint` | inferred at runtime | `0` |
| `minimap` | inferred at runtime | `Minimap(150, 150)` |
| `player` | inferred at runtime | `Car(Point(self.track.start_position.x, self.track.start_position.y), self.track.start_angle)` |
| `running` | inferred at runtime | `True` |
| `score` | inferred at runtime | `0` |
| `screen` | inferred at runtime | `pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))` |
| `state` | inferred at runtime | `GameState.RACING` |
| `timer` | inferred at runtime | `0` |
| `track` | inferred at runtime | `Track(SCREEN_WIDTH * 2, SCREEN_HEIGHT * 2)` |
**Methods:**
#### `handle_events(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/racing_game.py:830`
#### `update(self, steering: float, gas: float, dt: float)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** mutates instance state.
**Source:** `braindance/games/racing_game.py:861`
#### `render(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:928`
#### `_draw_hud(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:953`
#### `run(self)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:999`
### BackgroundGrid()
> Simple grid background to help visualize movement
**Source:** `braindance/games/racing_game.py:1027`
**Kind:** class. **Instantiated by:** braindance/games/racing_game.py:792 (named-call hint)
**Constructor:** `__init__(self, cell_size: int=100, color: Tuple[int, int, int]=DARK_GRAY)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `cell_size` | inferred at runtime | `cell_size` |
| `color` | inferred at runtime | `color` |
**Methods:**
#### `draw(self, surface: pygame.Surface, camera_offset: Point)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1035`
### Track()
> unclear — see source
**Source:** `braindance/games/racing_game.py:1053`
**Kind:** class. **Instantiated by:** braindance/games/racing_game.py:795 (named-call hint)
**Constructor:** `__init__(self, width: int, height: int)`
**Key attributes:**
| Name | Type | Assignment / meaning |
| --- | --- | --- |
| `collectibles` | inferred at runtime | `self._generate_collectibles()` |
| `height` | inferred at runtime | `height` |
| `obstacles` | inferred at runtime | `self._generate_obstacles()` |
| `segments` | inferred at runtime | `[TrackSegment(self.outer_points, self.inner_points, self.center_points)]` |
| `track_width` | inferred at runtime | `150` |
| `width` | inferred at runtime | `width` |
**Methods:**
#### `_generate_track(self)`
> Generate a simple oval track
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1078`
#### `_generate_obstacles(self)`
> Generate some obstacles around the track
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1127`
#### `_generate_collectibles(self)`
> Generate collectibles around the track
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1149`
#### `draw(self, surface: pygame.Surface, camera_offset: Point, dt: float=0.0)`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1167`
#### `is_point_on_track(self, point: Point) -> bool`
> Check if a point is on the track
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1199`
#### `_point_in_polygon(self, point: Point, polygon: List[Point]) -> bool`
> Check if point is inside a polygon using ray casting algorithm Credit: Modified from https://en.wikipedia.org/wiki/Point_in_polygon
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1209`
#### `distance_to_center(self, point: Point) -> float`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1230`
#### `check_lap(self, car_position: Point, last_checkpoint_idx: int) -> Tuple[bool, int]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1243`
#### `check_obstacle_collisions(self, car: Car) -> Optional[Obstacle]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1263`
#### `check_collectible_collisions(self, car: Car) -> List[Collectible]`
> unclear — see source
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1269`
#### `reset_collectibles(self)`
> Reset all collectibles to active state
> **Called by:** unresolved static dispatch. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1277`

## Functions
### `activate_boost(self)`
> Activate boost if available
> **Called by:** unclear — see source. **Side effects:** mutates instance state.
**Source:** `braindance/games/racing_game.py:1284`
### `get_distance_to_center(self, track: Track) -> float`
> Get distance to the center of the track
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1291`
### `draw(self, surface: pygame.Surface, camera_offset: Point)`
> Draw the car on the screen
> **Called by:** unclear — see source. **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1297`
### `main()`
> unclear — see source
> **Called by:** braindance/games/racing_game.py:1390 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/games/racing_game.py:1374`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| SCREEN_WIDTH=800,SCREEN_HEIGHT=600,FPS=60; arrows drive and Space boosts |
| Optional coin.wav,boost.wav,crash.wav relative to working directory |

## Data Shapes
- Point dataclass stores x,y; Track holds polygon points,checkpoints,Obstacle and Collectible objects

## Notes
- Later Track and BackgroundGrid class definitions replace earlier definitions; final functions monkey-patch Car methods.
- pygame.init executes on import; game itself is main-guarded; no neural or Gym environment integration.
