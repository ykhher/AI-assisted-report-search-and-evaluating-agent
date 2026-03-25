"""Entry point for the agent."""

from query_handler import generate_queries
from search_api import SearchAPI
from parser import parse_search_results
from filter import filter_results
from duplicate import deduplicate
from ranking import rank_results


def main():
    query = input("Search query: ").strip()
    queries = generate_queries(query)

    api = SearchAPI()

    all_results = []
    for q in queries:
        raw = api.search([q])
        parsed = parse_search_results(raw)
        all_results.extend(parsed)

    filtered = filter_results(all_results)
    unique = deduplicate(filtered)
    ranked = rank_results(unique)

    print("Final candidates:", len(unique))
    for item in ranked:
        print(item)


if __name__ == "__main__":
    main()
