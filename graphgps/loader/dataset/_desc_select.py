"""Descriptor-column selection for the biodeg loaders (Phase 2 study).

Resolves WHICH descriptor columns a run uses, from three optional specs, and
derives a short stable hash of the resolved selection so each subset gets its
own processed cache (no silent collisions on data_stdesc.pt).

Selection precedence:
    desc_columns (explicit exact names)  >  desc_include / desc_exclude
    (substring match)  >  all columns (default).
`desc_exclude` is always applied last (also on top of an explicit/included set).
"""
import hashlib


def select_descriptor_columns(all_columns, include=None, exclude=None,
                              columns=None):
    """Return the ordered subset of ``all_columns`` per the selection spec.

    Args:
        all_columns: full ordered list of descriptor column names (manifest).
        include: substrings; keep a column if it contains ANY of them.
        exclude: substrings; drop a column if it contains ANY of them.
        columns: explicit exact names; if given, select exactly those (kept in
            ``all_columns`` order), then apply ``exclude``.
    Returns:
        list[str]: selected column names, subset of all_columns, original order.
    Raises:
        ValueError: empty selection, or ``columns`` references unknown names.
    """
    include = list(include or [])
    exclude = list(exclude or [])
    columns = list(columns or [])
    all_set = set(all_columns)

    if columns:
        unknown = [c for c in columns if c not in all_set]
        if unknown:
            raise ValueError(
                f"desc_columns references columns not in the dataset: {unknown}"
            )
        wanted = set(columns)
        selected = [c for c in all_columns if c in wanted]
    elif include:
        selected = [c for c in all_columns if any(s in c for s in include)]
    else:
        selected = list(all_columns)

    if exclude:
        selected = [c for c in selected if not any(s in c for s in exclude)]

    if not selected:
        raise ValueError(
            f"descriptor selection produced 0 columns "
            f"(include={include}, exclude={exclude}, columns={columns})."
        )
    return selected


def selection_tag(selected_columns):
    """Short, stable, order-sensitive hash of the resolved column list.

    Used to key the processed cache so different subsets never collide. Order
    matters (the model sees columns in this order), so the hash is over the
    ordered, newline-joined names.
    """
    digest = hashlib.sha1('\n'.join(selected_columns).encode('utf-8'))
    return digest.hexdigest()[:8]
