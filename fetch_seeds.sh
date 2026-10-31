#!/bin/bash
# Biome's seed images (Overworld's repository, pinned commit) into ./seeds. Not redistributed here.
set -eu
cd "$(dirname "$0")"; mkdir -p seeds
for n in abandoned_forest_cabin alien_command_center ancient_standing_stones crystal_desert_blade default \
  desert_outpost_hangar enchanted_swamp_torch frozen_crystal_cavern frozen_valley_sniper highland_castle_loch \
  mountain_ruins_sword neon_alley_night red_canyon_spear shattered_cockpit_nebula shipwreck_shore_revolver \
  snowy_forest_tracks stormy_countryside_rifle sunken_city_depths; do
  curl -sfL -o seeds/$n.jpg "https://raw.githubusercontent.com/Overworldai/Biome/14343a6/seeds/$n.jpg"
done
ls seeds | wc -l
