"""
File that handles communication with the user.
"""
import argparse
import algorithms_on_graphs as lib


ALGORITHMS = {
    'euler-cycle': ('Performs the Euler cycle algorithm on a graph.', lib.euler_cycle),
    'hamiltonian-cycle': ('Checks whether the graph has Hamiltonian cycle.', lib.check_for_ham),
    'bipartite-graph': ('Checks whether the graph is bipartite.', lib.bipartite_graph_check),
    'isomorphic-graphs': ('Checks if two graphs are isomorphic.',\
lib.if_graphs_are_isomorphic),
    'coloring-graphs': ('Colors the graph using the provided colors.', lib.write_colour),
    'hamiltonian-visualization': ('Visualize the search of Hamiltonian path', lib.tkinter_window)
}

def main():
    """
    Core of communication with user via argparse
    """

    parser = argparse.ArgumentParser(description = 'Process a graph with a selected algorithm.')
    parser.add_argument('file', help = 'Path to the .dot file with the graph.')
    parser.add_argument('--add-file', help = 'Path to the .dot file with\
an additional graph (required for isomorphism checking).')
    parser.add_argument('algorithm', help = 'Name of the algorithm to apply to the graph.')
    parser.add_argument('--file-out', help = 'File to write the output\
(required for coloring only).', type = str)
    parser.add_argument('--colors', help = 'List of colors for graph coloring\
(required for coloring only).', type = str)
    parser.add_argument('--visualization-for-coloring', help = 'Parameter for coloring graph in 3 colours', type = bool)

    args = parser.parse_args()

    if args.algorithm in ALGORITHMS:
        print(f"Performing '{args.algorithm}' algorithm on {args.file}.")
        description, alg = ALGORITHMS[args.algorithm]
        print(f"Description of the algorithm: {description}")

        graph = lib.read_file(args.file)
        if args.algorithm == 'isomorphic-graphs':
            if not args.add_file:
                print("'isomorphic-graphs' requires an additional file.\
Use the --add-file option.")
                return
            graph_add = lib.read_file(args.add_file)
            result = alg(graph, graph_add)
            print(f'Result of an algorithm: {result}')
        elif args.algorithm == 'coloring-graphs':
            if not args.file_out or not args.colors:
                print("'coloring-graphs' requires --file-out,\
--colors options(ex. red,yellow,blue) and paremeter for visualizing the coloring")
                return
            colors = [c for c in args.colors.split(",")]
            result = alg(args.file, args.file_out, colors, args.visualization_for_coloring)
        elif args.algorithm == 'hamiltonian-visualization':
            result = alg()
        else:
            result = alg(graph)
            print(f'Result of an algorithm: {result}')
    else:
        print(f"Unknown algorithm '{args.algorithm}'. Choose from the available options:")
        for name, (desc, _) in ALGORITHMS.items():
            print(f"  {name}: {desc}")


if __name__ == "__main__":
    main()
