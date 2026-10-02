"""The declared reading of the ring tile sweep (`gdn_ring_tile_sweep.py`): its configuration and
threshold rule, written down before the sweep ran (evidence README, "Ring-writing verify tiles
by batch: declared reading"). No torch import, so CPU tools and tests can apply the rule to a
saved report.
"""

from __future__ import annotations

from typing import Any

# The configuration the threshold rule was declared on. A run that differs in any of these
# gets no threshold.
DECLARED: dict[str, Any] = {
    'blocks': [16, 8],
    'batches': [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64],
    'tiles': [4, 8, 16, 32],
    'layers': 24,
    'iters': 50,
    'repeats': 4,
}
# Lists that name a set of grid values, compared without regard to order.
_SETS = ('blocks', 'batches', 'tiles')


def report_config(report: dict[str, Any]) -> dict[str, Any]:
    """The configuration a saved sweep report records: the grid as measured (from its rows),
    and the layer, iteration and repeat counts. If any row holds a different number of
    repeats than the report states, the repeat count is the list of counts found, which
    never matches the declared one."""
    rows = report['rows']
    repeats = {len(r['us_per_layer_by_repeat']) for r in rows}
    return {
        'blocks': sorted({r['T'] for r in rows}),
        'batches': sorted({r['N'] for r in rows}),
        'tiles': sorted({r['BV'] for r in rows if r['path'] == 'ring'}),
        'layers': report['shape']['layers'],
        'iters': report['iters'],
        'repeats': report['repeats'] if repeats == {report['repeats']} else sorted(repeats),
    }


def config_differences(config: dict[str, Any]) -> list[str]:
    """Every declared parameter the given configuration does not match, as readable strings."""
    differences = []
    for key, declared in DECLARED.items():
        value = config.get(key)
        if key in _SETS:
            same = isinstance(value, list) and sorted(value) == sorted(declared)
        else:
            same = value == declared
        if not same:
            differences.append(f'{key}: {value} (declared {declared})')
    return differences


def declared_threshold(
    rows: list[dict[str, Any]], blocks: list[int], batches: list[int]
) -> dict[str, Any]:
    """The batch threshold N* by the rule declared before the sweep ran (evidence README).

    Tile 4 wins at (T, N) when it is faster than tile 32 in every repeat and the gap
    between their medians exceeds the larger of the two tiles' repeat ranges. N*_T is
    the largest grid batch such that tile 4 wins at it and at every smaller grid batch
    (0 if it loses at the smallest), and N* is the smaller of N*_T over the blocks, since
    the launch-config selection sees the batch but not the block length.
    """
    result: dict[str, Any] = {'wins': {}, 'n_star_by_block': {}}
    for T in blocks:
        star = 0
        prefix = True
        for n in batches:
            by_bv = {
                r['BV']: r for r in rows if r['T'] == T and r['N'] == n and r['path'] == 'ring'
            }
            narrow, wide = by_bv[4], by_bv[32]
            spread = max(
                narrow['us_per_layer_range'][1] - narrow['us_per_layer_range'][0],
                wide['us_per_layer_range'][1] - wide['us_per_layer_range'][0],
            )
            every = all(
                a < b
                for a, b in zip(
                    narrow['us_per_layer_by_repeat'], wide['us_per_layer_by_repeat'], strict=True
                )
            )
            gap = wide['us_per_layer_median'] - narrow['us_per_layer_median']
            wins = every and gap > spread
            result['wins'][f'T{T}_N{n}'] = wins
            prefix = prefix and wins
            if prefix:
                star = n
        result['n_star_by_block'][f'T{T}'] = star
    result['n_star'] = min(result['n_star_by_block'].values())
    return result


def threshold_for_config(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """The declared threshold if the run used the declared configuration; otherwise none,
    with every parameter that differs."""
    differences = config_differences(config)
    if differences:
        return {
            'n_star': None,
            'reason': 'the run differs from the declared configuration, '
            'so the declared rule does not apply',
            'differences': differences,
        }
    return declared_threshold(rows, DECLARED['blocks'], DECLARED['batches'])
