import networkx as nx


def bfs_layers(G, sources):
    """Returns an iterator of all the layers in breadth-first search traversal, allowing nodes to be revisited."""
    assert nx.is_directed_acyclic_graph(G), "Graph must be a directed acyclic graph (DAG)."

    if sources in G:
        sources = [sources]

    current_layer = list(sources)

    for source in current_layer:
        if source not in G:
            raise nx.NetworkXError(f"The node {source} is not in the graph.")

    while current_layer:
        yield current_layer
        next_layer = []
        seen = set()
        for node in current_layer:
            for child in G[node]:
                if child not in seen:
                    seen.add(child)
                    next_layer.append(child)
        current_layer = next_layer


def get_depths_from_bfs_layers(G, sources):
    depth = {}
    for layer_index, layer in enumerate(bfs_layers(G, sources)):
        for node in layer:
            depth[node] = layer_index
    return depth
