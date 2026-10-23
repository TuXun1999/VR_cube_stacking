# Robot tabletop map

`islands.png` is an unchanged copy of OpenUp Cube Stack's default island map
from `openup_apps/cube_stack/assets/islands.png`. OpenUp created this image
with its built-in image-generation tool for the Cube Stack application.

`table_map.obj` is a two-triangle visual surface. Its UV coordinates map image
right to Genesis +Y and image down to Genesis +X, matching the cube importer.
The simulation fits it to the workspace table and reserves the left 12% for
the palette margin used by OpenUp. The surface has no collisions.
