# CFP camera convention

The selected coordinate contract is named `CFP_TURNTABLE_V1`.

The character remains fixed and upright:

- local `+Z` is up;
- local `+Y` is the character's forward direction;
- local `+X` is the character's right side.

Azimuth describes the camera position around the stationary character, viewed
from above. Positive azimuth rotates counterclockwise from the front camera
position.

| Azimuth | Camera position | Image description |
| ---: | --- | --- |
| 000° | `+Y` | Front |
| 045° | between `+Y` and `-X` | Front-left three-quarter |
| 090° | `-X` | Character's left profile |
| 135° | between `-X` and `-Y` | Rear-left three-quarter |
| 180° | `-Y` | Rear |
| 225° | between `-Y` and `+X` | Rear-right three-quarter |
| 270° | `+X` | Character's right profile |
| 315° | between `+X` and `+Y` | Front-right three-quarter |

Default values:

```text
Elevation: 0°
Roll: 0°
```

The camera looks toward the object origin. The required prompt contract is:

> Rotate only the camera. The object has not changed.

Names use zero-padded azimuths:

```text
camera_000
camera_090
camera_180
camera_270
```

Artifact names put the subject or component first:

```text
character_camera_000.png
faceplate_camera_090.png
```

This mapping must not be silently reversed. A future convention change requires
a new convention ID and an explicit migration.

