


import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    queue = [(0, start)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))

    return distances
graph = {
    'A': {'B': 3, 'C': 2},
    'B': {'A': 3, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 7},
    'D': {'B': 5, 'C': 7}
}

start = 'A'
x=dijkstra(graph,start)
print('the shortest path is',x)

def shortest_path_dp(graph, start, end):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node].items():
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight

    path = [end]
    while end != start:
        for node in graph:
            for neighbor, weight in graph[node].items():
                if distances[node] + weight == distances[end]:
                    path.append(node)
                    end = node
                    break

    return distances[end], list(reversed(path))
end='B'

print ('the shortest PATH between',start,'and',end ,'is',shortest_path_dp(graph,start,end))
import math

def floyd_warshall(graph):
    

    V = len(graph)
    dist = graph.copy()  # Create a copy of the adjacency matrix

    # Iterate through all possible intermediate vertices
    for k in range(V):
        for i in range(V):
            for j in range(V):
                
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    
    for i in range(V):
        if dist[i][i] < 0:
            return None  
    return dist

graph = [[0, 3, math.inf, 5],
         [2, 0, 4, math.inf],
         [math.inf, 1, 0, 6],
         [math.inf, math.inf, 2, 0]]

shortest_paths = floyd_warshall(graph)
print(shortest_paths)

def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=" ")  # Print the node as it's visited

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

print("DFS traversal starting from node A:")
dfs(graph, 'A')

def bfs(graph, start):
    visited = set()
    queue = [start]

    while queue:
        node = queue.pop(0)  # Dequeue the first node
        if node not in visited:
            visited.add(node)
            print(node, end=" ")  # Print the node as it's visited
            queue.extend(graph[node])  # Enqueue all unvisited neighbors

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

print("\nBFS traversal starting from node A:")
bfs(graph, 'A')



def prim_mst(graph):
    mst = []
    visited = set()
    start_node = next(iter(graph))

    visited.add(start_node)
    edges = [(cost, start_node, neighbor) for neighbor, cost in graph[start_node]]

    heapq.heapify(edges)

    while edges:
        cost, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            mst.append((u, v, cost))
            for neighbor, cost in graph[v]:
                if neighbor not in visited:
                    heapq.heappush(edges, (cost, v, neighbor))

    return mst
graph = {
    'A': [('B', 3), ('C', 2)],
    'B': [('A', 3), ('C', 1), ('D', 5)],
    'C': [('A', 2), ('B', 1), ('D', 7)],
    'D': [('B', 5), ('C', 7)]
}
print('\n',prim_mst(graph))


def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    x_root = find(parent, x)
    y_root = find(parent, y)

    if rank[x_root] < rank[y_root]:
        parent[x_root] = y_root
    elif rank[x_root] > rank[y_root]:
        parent[y_root] = x_root
    else:
        parent[y_root] = x_root
        rank[x_root] += 1

def kruskal_mst(graph):
    mst = []
    edges = []
    parent = {}
    rank = {}

    for node in graph:
        parent[node] = node
        rank[node] = 0
        for neighbor, cost in graph[node]:
            edges.append((cost, node, neighbor))

    edges.sort()

    for edge in edges:
        cost, u, v = edge
        if find(parent, u) != find(parent, v):
            mst.append((u, v, cost))
            union(parent, rank, u, v)

    return mst
print('\n',kruskal_mst(graph))