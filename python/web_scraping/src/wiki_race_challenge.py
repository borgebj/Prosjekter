from collections import deque
import time
from typing import Callable, List

from filter_urls import find_articles
from requesting_urls import get_html


def average_time(
    function: Callable,
    *args,
    calls: int = 1
) -> tuple[List[str], float]:
    times = []
    paths = []

    for _ in range(calls):
        t0 = time.perf_counter()
        paths.append(function(*args))
        t1 = time.perf_counter()
        times.append(t1-t0)

    return paths[0], sum(times) / len(times)


def countdown(sec):
    time.sleep(1)
    while sec > 0:
        print(f'[{sec}]', end=" ")
        sec -= 1
        time.sleep(1)


def BFS_shortest_path(start: str, goal: str) -> List[str]:
    """Breadth-first search algorithm, for traversing a graph

    Traverses a link and all its sub-links to find a path
    between first url and second url

    Arguments:
        start (string) : string containing URL to start on
        goal (string) : string containing URL to end up on
    Returns:
        new_path (list) : a list containing the path from start to goal in URL-form
    """
    queue = deque([start])
    visited = {start}
    parent = {start: None}

    while queue:
        node = queue.popleft()

        # goes through all wikipedia articles from this page
        for article in find_articles(get_html(node), all_lan=False):
            if article in visited:
                continue

            # mark article as vied, save parent / where it came from
            visited.add(article)
            parent[article] = node

            if article == goal:
                path = []
                current = goal

                # reconstruct path following parent backwards
                while current is not None:
                    path.append(current)
                    current = parent[current]

                return path[::-1]

            queue.append(article)

    return []


def find_path(start: str, finish: str, calls: int) -> List[str]:
    """Find the shortest path from `start` to `finish`

    Uses Breadth-first search algorithm to search through all articles of each url,
    to then first to shorten the path to the end-article

    Arguments:
      start (str): wikipedia article URL to start from
      finish (str): wikipedia article URL to stop at
      calls (int): how many calls for the timing function
    Returns:
      urls (list[str]):
        List of URLs representing the path from `start` to `finish`.
        The first item should be `start`.
        The last item should be `finish`.
        All items of the list should be URLs for wikipedia articles.
        Each article should have a direct link to the next article in the list.
    """

    print("\nStarting wiki-race in ...")
    countdown(3)
    print("\n\n- Search has begun -")

    # starts timing
    t0 = time.perf_counter()
    time_start = time.strftime("%H:%M:%S", time.localtime())

    path, avg = average_time(
        BFS_shortest_path,
        start,
        finish,
        calls=calls)

    time_end = time.strftime("%H:%M:%S", time.localtime())
    t1 = time.perf_counter()
    # ends timing

    # inform of output
    print("\n"+("-"*100))
    print("Shortest path:")
    [print(f" -> {article}") for article in path]
    print(f'Total time: {t1-t0:.2f}s on {calls} runs with {avg:.2f}s average')
    print(f'Time at start: {time_start}')
    print(f'Time at end: {time_end}')
    print("-"*100)

    assert path[0] == start
    assert path[-1] == finish

    return path


if __name__ == "__main__":
    # default
    # start = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    # finish = "https://en.wikipedia.org/wiki/Peace"

    # shorter
    # start = 'https://en.wikipedia.org/wiki/J._K._Rowling'
    # finish = 'https://en.wikipedia.org/wiki/World_War_II'

    # shortest
    start = "https://en.wikipedia.org/wiki/Pentacora"
    finish = "https://en.wikipedia.org/wiki/Adolf_Hitler"
    calls = 1

    find_path(start, finish, calls)
