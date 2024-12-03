"""Computer project from discrete mathematics"""

import time

#  yulian ham cycle imports
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import ast

def read_file(file_name: str, extra_list = False) -> list[tuple] | list[set]:
    """
    The function reads the file .dot and makes 
    the list of tuples of neighbour vertices.

    :param file_name: str, the name of the file.
    :return: list[tuple] | list[set], the list for graph.
    list of tuples for directed graph.
    list of sets for not directed graph.
    """
    with open(file_name, 'r', encoding='utf-8') as file:
        first = file.readline()
        res = []
        verts_set = set()
        if "digraph" in first:
            spl = "->"
        else:
            spl = "--"
        for line in file:
            if spl in line:
                verts = line.strip().split(spl)
                i = 0
                verts_set.add(verts[0].strip())
                while i != len(verts) - 1:
                    verts_set.add(verts[i + 1].strip())
                    res.append((verts[i].strip(), verts[i + 1].strip()))
                    i += 1
        verts_list = sorted(list(verts_set))
        res1 = [(verts_list.index(i[0]) + 1, verts_list.index(i[1]) + 1) for i in res]
        res2 = []
        # edgecase check 
        for i in res1:
            if set(i) not in res2:
                res2.append(set(i))
        if extra_list is True:
            return [res1, verts_list] if spl == "->" else [res2, verts_list]
        return res1 if spl == "->" else res2

def convert_to_directed(graph: list[set]) -> list[tuple]:
    """
    Converts undirected graphs into directed.

    :param graph: list[set], undirected graph
    :return: list[tuple], directed graph

    >>> graph = [{'a', 'b'}, {'b', 'c'}, {'c', 'a'}]
    >>> sorted(convert_to_directed(graph)) == \
sorted([('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'b'), ('c', 'a'), ('a', 'c')])
    True
    >>> sorted(convert_to_directed([{'a', 'b'}, {'b', 'c'}, {'c', 'd'}, {'d', 'a'}])) == \
sorted([('b', 'a'), ('a', 'b'), ('b', 'c'), ('c', 'b'), ('d', 'c'), ('c', 'd'), \
('d', 'a'), ('a','d')])
    True
    """
    output = []
    for edge in graph:
        ver1, ver2 = edge
        output.extend([(ver1, ver2), (ver2, ver1)])

    return output


def euler_cycle(graph: list[tuple | set]) -> list[list[str]]:
    """
    Check whether 

    :param graph: list[tuple] or list[set], list of tuples of letters that symbolyize verteces.
    :return: list[list[str]],  list of all possible euler cycles.

    >>> graph = [('a', 'b'), ('c', 'b'), ('d', 'c'), ('d', 'a'), \
                 ('b', 'd'), ('b', 'd')]
    >>> euler_cycle(graph)
    "This graph isn't strongly connected"

    >>> graph1 = [('a', 'b'), ('b', 'a'), ('c', 'd'), ('d', 'c'), \
                 ('a', 'd'), ('d', 'a'), ('b', 'd'), ('d', 'b')]
    >>> euler_cycle(graph1)
    'There is no Euler cycle for this graph'

    >>> graph2 = [('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'b'), ('c', 'a'), ('a', 'c'), \
                 ('c', 'd'), ('d', 'c'), ('d', 'e'), ('e', 'd'), ('e', 'c'), ('c', 'e')]
    >>> euler_cycle(graph2)
    [['a', 'b', 'c', 'd', 'e', 'c'], ['a', 'b', 'c', 'e', 'd', 'c']]

    >>> graph3 = [{'a', 'b'}, {'b', 'c'}, {'c', 'd'}, {'d', 'a'}]
    >>> euler_cycle(graph3)
    [['a', 'b', 'c', 'd']]

    >>> graph4 = [{'a', 'b'}, {'b', 'c'}, {'c', 'd'}, {'d', 'a'}, {'a', 'e'}, {'b', 'e'}, \
{'c', 'e'}, {'d', 'e'}, {'f', 'd'}, {'f', 'c'}, {'g', 'a'}, {'g', 'b'}]
    >>> ['a', 'e', 'd', 'f', 'c', 'e', 'b', 'c', 'd', 'a', 'b', 'g'] in euler_cycle(graph4)
    True
    """
    if not graph:
        return "This graph is empty"
    # deepcopy to be on the safe side
    p_graph = graph.copy()
    length = len(graph)

    if flag := isinstance(graph[0], set):
        p_graph = []
        for edge in graph:
            ver1, ver2 = edge
            p_graph.extend([(ver1, ver2), (ver2, ver1)])

    # calculate_way writes all possible cycles here
    all_cycles = []

    def calculate_way(graph: list[tuple], position: str, way: str, r_p: tuple) -> list[tuple] | str:
        """
        Takes a graph and current position. Check where you can go from your current position
        and creates all these possible ways.
        """
        graph = graph.copy()
        graph.remove(r_p)
        if flag:
            graph.remove(r_p[::-1])
        for pair in graph:
            if pair[0] == position:
                if pair[1] == vertex and len(way) == length:
                    all_cycles.append(way)
                calculate_way(graph, pair[1], way+pair[1], pair)

    # This is a vertix from which we move
    vertex = p_graph[0][0] if p_graph[0][0] < p_graph[0][1] else p_graph[0][1]
    calculate_way(p_graph + ['*', '*'], vertex, vertex, '*')

    return [list(el) for el in sorted(set(all_cycles))] if all_cycles else 'There is no euler cycle for this graph'



def permute(nodes):
    """
    Generate all permutations of the given list.
    >>> permute([1,2,3])
    [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    """
    if len(nodes) == 1:
        return [nodes]

    perm = []
    n_len = len(nodes)

    for i in range(n_len):
        current = nodes[i]
        remaining = nodes[:i] + nodes[i+1:]

        for p in permute(remaining):
            perm.append([current] + p)

    return perm

def check_for_ham(graph: list[tuple]) -> list | str:
    """
    Checks a graph if it has a hamiltonian cycle or not

    :param graph: list[tuple], a list of tuples, where the first number
    is the vertice, and the second is the connection

    :return: returns either a list of the vertices of the hamiltonian
    cycle, or says that it does nto exist

    # UNORIENTED
    >>> g = [{1, 2}, {2, 3}, {3, 1}]
    >>> check_for_ham(g)
    [1, 2, 3, 1]

    >>> g1 = [{1, 4}, {4, 3}, {3, 5}, {5, 1}]
    >>> check_for_ham(g1)
    [1, 4, 3, 5, 1]

    >>> g2 = [{1, 5}, {5, 4}, {4, 3}, {3, 2}, {2, 1}]
    >>> check_for_ham(g2)
    [1, 5, 4, 3, 2, 1]

    >>> g3 = [{1, 2}, {2, 4}, {5, 1}]
    >>> check_for_ham(g3)
    'There is no Hamiltonian cycle for this graph'

    >>> g4 = [{1, 2}, {2, 3}, {3, 2}, {2, 1}]
    >>> check_for_ham(g4)
    'There is no Hamiltonian cycle for this graph'

    # ORIENTED
    >>> g_oriented = [(1, 2), (2, 3), (3, 1)]
    >>> check_for_ham(g_oriented)
    [1, 2, 3, 1]

    >>> g1_oriented = [(1, 4), (4, 3), (3, 5), (5, 1)]
    >>> check_for_ham(g1_oriented)
    [1, 4, 3, 5, 1]

    >>> g2_oriented = [(1, 5), (5, 4), (4, 3), (3, 2), (2, 1)]
    >>> check_for_ham(g2_oriented)
    [1, 5, 4, 3, 2, 1]

    >>> g3_oriented = [(1, 2), (2, 4), (5, 1)]
    >>> check_for_ham(g3_oriented)
    'There is no Hamiltonian cycle for this graph'

    >>> g4_oriented = [(1, 2), (2, 3), (3, 2), (2, 1)]
    >>> check_for_ham(g4_oriented)
    'There is no Hamiltonian cycle for this graph'
    """

    if len(graph) < 3:
        return "To have a Hamiltonian graph you need at least 3 edges!"

    # Check what type of graph
    unoriented = None

    if isinstance(graph[0], set):
        unoriented = True

    con_dic = {}


    for tup in graph:
        tup = list(tup)
        # make connections for first vertices
        if tup[0] in con_dic:
            con_dic[tup[0]].append(tup[1])
        elif tup[0] not in con_dic:
            con_dic[tup[0]] = [tup[1]]
        if unoriented:
            # Give second vertice con to first
            if tup[1] in con_dic:
                con_dic[tup[1]].append(tup[0])
            elif tup[1] not in con_dic:
                con_dic[tup[1]] = [tup[0]]

    # Check dic for duplicates
    for key1, val1 in con_dic.items():
        con_dic[key1] = list(set(val1))

    # Start extremely cool checking logic (please help me)
    all_keys = [k for k in con_dic]

    # Make all possible
    all_possible = permute(all_keys)

    # Add the first to the end of each list
    for a in all_possible:
        a.append(a[0])

    test_len = len(all_possible[0])
    # Check which of all are good
    for path in all_possible:
        test = 1
        for index in range(1, test_len):
            if path[index] in con_dic[path[index - 1]]:
                test += 1
            elif path[index] not in con_dic[path[index - 1]]:
                break

        if test == test_len:
            return path


    return "There is no Hamiltonian cycle for this graph"

#pylint: disable=all

def parse_input(text):
    """
    Changes string text into actual data
    """
    try:
        return ast.literal_eval(text)
    except Exception as e:
        return None

# display of results
def display_results(permutations, correct_path, indx=0):
    """
    Displays the results of the calculations in the scrollable text box
    """
    output_box.tag_configure("green_text", foreground="green")

    # When all results are displayed
    if indx >= len(permutations):
        output_box.insert(tk.END, "Checking complete.\n")
        if correct_path:
            output_box.insert(tk.END, f"The correct Hamiltonian cycle is: {correct_path}\n")
        else:
            output_box.insert(tk.END, "No Hamiltonian cycle found.\n")
        output_box.see(tk.END)
        return None

    path = permutations[indx]
    valid_cycle = (path == correct_path)
    result = "Correct" if valid_cycle else "Wrong"

    output_box.insert(tk.END, f"Checking: {path} - ")

    if valid_cycle:
        output_box.insert(tk.END,f"{result}\n", "green_text")
    else:
        output_box.insert(tk.END,f"{result}\n")

    output_box.see(tk.END)
    root.after(50, lambda: display_results(permutations, correct_path, indx + 1))





def on_enter(event):
    """
    Defines what has to be done when enter is pressed in the main text box,
    in this case call the deploy_results function
    """
    input_text = input_box.get()
    input_box.delete(0, tk.END)
    graph = parse_input(input_text)
    if graph:
        output_box.delete(1.0, tk.END)
        output_box.insert(tk.END, "Processing...\n")
        root.update()

        correct_path_or_message = check_for_ham(graph)
        permutations = permute([key for key in {k for edge in graph for k in edge}])

        # Closes cycle
        for p in permutations:
            p.append(p[0])

        # No hamiltonian, all are wrong
        if isinstance(correct_path_or_message, str):
            output_box.insert(tk.END, correct_path_or_message + "\n")
            display_results(permutations, [])

        # Yes Hamiltonian, checks which one is correct
        else:
            display_results(permutations, correct_path_or_message)
    else:
        output_box.insert(tk.END, "Invalid input format. Please try again.\n")
    output_box.see(tk.END)

def tkinter_window():

    global root
    root = tk.Tk()

    # Tkinter window setup

    root.title("Hamiltonian Cycle Checker")
    root.geometry("800x1000")

    font_style = ("Arial", 14)
    label = tk.Label(root, text="Enter graph vertices as a list of sets or tuples (for example: [(1, 2), (2, 3), (3, 1)]):", font=font_style)
    label.pack(pady=10)

    global input_box
    input_box = tk.Entry(root, font=("Arial", 14), width=80)
    input_box.pack(pady=10)

    global output_box
    output_box = ScrolledText(root, font=("Arial", 12), width=90, height=50, state='normal')
    output_box.pack(pady=10)

    input_box.bind("<Return>", on_enter)

    root.mainloop()

#pylint: enable=all

def to_matrix(graph_list: list[tuple | set]) -> list[list]:
    """
    The function makes the matrix for the graph.

    :param graph_list: list[tuple | set], the graph.
    :return: list[list], matrix.
    >>> to_matrix([(1, 2), (2, 3), (3, 4), (1, 5), (5, 3)])
    [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]]
    """
    if len(graph_list) > 0 and isinstance(graph_list[0], set) is True:
        gr = []
        for el in graph_list:
            e = list(el)
            gr.append((e[0], e[1]))
            gr.append((e[1], e[0]))
        graph_list = gr
    leng = max(max(i[0], i[1]) for i in graph_list)
    return [[1*((i+1, j+1) in graph_list) for i in range(leng)] for j in range(leng)]

def to_symetric(matrix: list[list]) -> list[list]:
    """
    The function returns the symetric relation.

    :param matrix: list[list], the graph's matrix.
    :return: list[list], the symetric relation.
    >>> to_symetric([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 1], \
[0, 0, 1, 0, 0], [1, 0, 0, 0, 0]])
    [[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 0], [1, 0, 1, 0, 0]]
    """
    leng = len(matrix)
    return [[matrix[i][j] | matrix[j][i] for i in range(leng)] for j in range(leng)]

def approp(cur: int, graph: list[list[int]], colours: list[int], colour: int) -> bool:
    """
    The function checks whether we can assign colour to the vertice.

    :param cur: int, the number of current vertice.
    :param graph: list[list[int]], the matrix of the graph.
    :param colours: list[int], the list with colours' numbers assigned to each vertice.
    :param colour: int, the colour's number wanted to be assigned.
    :return: bool, if the colour is appropriate - True, else - False.
    """
    for i in range(cur):
        if graph[cur][i] and colour == colours[i]:
            return False
    return True

def visualizing(file_name: str, cur: int, colour: int, st_col: list[str], ver_list: list[str]) -> None:
    """
    The function helps visualize the algorithm's work. It writes the file every time the colour is being checked
    adding for the output file the line with new description(colour). The file exists from the very begining and 
    from the begining obtains the same information the input file has.

    :param file_name: str, the output file.
    :param cur: int, the number of the vertice.
    :param colour: int, the number of the colour.
    :param st_col: list[str], the input list with colour names.
    :param ver_list: list[str], the initial list of vertices names.
    """
    content = []
    time.sleep(0.89)
    with open(file_name, "r", encoding="utf-8") as file:
        content = file.readlines()
    with open(file_name, 'w+', encoding='utf-8') as file:
        for line in content[:-1]:
            file.write(line)
        file.write(f'\t{ver_list[cur]} [shape = circle style = filled \
color="{st_col[colour - 1]}"]\n')
        file.write("}")

def colouring(graph: list[list[int]], s: int, colours: list[int], st_col: list[str], \
              cur: int, visualize: bool, file_out: str, ver_list: list[str]) -> bool | list[int]:
    """
    The function is "colouring" the graph.

    :param graph: list[list[int]], the matrix of the graph.
    :param s: int, the number of vetices.
    :param colours: list[int], the list with colours' numbers assigned to each vertice.
    :param st_col: list[str], the input list with colour names.
    :param cur: int, the number of curren vertice.
    :param visualize: bool, wheather the visualisation is needed.
    :param file_out: str, the output file.
    :param ver_list: list[str], the initial list of vertices names.
    :return: bool(False if we can't color the graph) or the changed list colours.
    """
    if cur == s:
        return True

    for i in range(1, len(st_col) + 1):
        if visualize is True:
            visualizing(file_out, cur, i, st_col, ver_list)
        if approp(cur, graph, colours, i):
            colours[cur] = i

            if colouring(graph, s, colours, st_col, cur+1, visualize, file_out, ver_list):
                return colours
    return False

def get_colour_seq(file_name: str, colour_list: list[str], visualize: bool, file_out: str) -> str:
    """
    The function returns the result of colouring and the text if not possible.

    :param file_name: str, the name of .dot file.
    :param colour_list: list[str], the chosen colours' names (only 3).
    :param visualize: bool, wheather the visualisation is needed.
    :param file_out: str, the output file.
    :return: str, the colour sequence in string with white spaces.
    """
    rel = read_file(file_name, True)
    matrix = to_symetric(to_matrix(rel[0]))
    s = len(matrix)
    chosen_colours = [0]*s
    if colouring(matrix, s, chosen_colours, colour_list, 0, visualize, file_out, rel[1]):
        chosen_colours = colouring(matrix, s, chosen_colours, colour_list, 0, visualize, file_out, rel[1])
        result = " ".join(map(lambda x: colour_list[x-1], chosen_colours))
    else:
        result = "The colouring is imposible."
    return [result, rel[1]]

def write_colour(file_in: str, file_out: str, colours: list[str], visualize = False) -> None:
    """
    The function writes the coloured graph into the file.

    :param file_in: str, the name of input .dot file.
    :param file_out: str, the name of output .dot file.
    :param colours: str, the chosen colours' names (only 3).
    :param visualize: bool, wheather the visualisation is needed.
    :return: None
    """
    if visualize is True:
        content = []
        with open(file_in, 'r', encoding='utf-8') as file:
            content = file.readlines()
        with open(file_out, "w+", encoding='utf-8') as file:
            for line in content:
                file.write(line)
    sequence = get_colour_seq(file_in, colours, visualize, file_out)
    if sequence[0] == "The colouring is imposible.":
        print(sequence[0])
    else:
        sequence, verts = sequence[0], sequence[1]
        with open(file_out, "w+", encoding='utf-8') as file, \
        open(file_in, 'r', encoding='utf-8') as f:
            for line in f.readlines()[:-1]:
                file.write(line)
            for i, ver in enumerate(verts):
                file.write(f'\t{ver} [shape = circle style = filled \
color="{sequence.split()[i]}"]\n')
            file.write("}")
# write_colour("graph.dot", "coloured.dot", ["red", "blue", "green"], True)

def bipartite_graph_check(graph: list[tuple] | list[set])-> bool:
    """
    Function checks if a given graph is bipartite.

    A graph is bipartite if its vertices can be divided into two disjoint sets 
    such that no two vertices within the same set are adjacent.
    
    Using a breadth-first search (BFS) approach 
    to check whether the graph satisfies this property
    by coloring vertices into 2 colors.

    :param graph: list[tuple] or list[set], given graph
    :return: bool, function returns True if graph is bipartite, 
    returns False when it is not.

    >>> graph = [{'a', 'b'}, {'b', 'c'}, {'c', 'e'}, {'c', 'd'}, {'a', 'd'}, {'a', 'e'}]
    >>> bipartite_graph_check(graph)
    True
    >>> graph = [{'a', 'b'}, {'a', 'c'}, {'a', 'd'}, {'b', 'c'}, {'b', 'd'}, {'c', 'd'}]
    >>> bipartite_graph_check(graph) # example for complete graph with more than 2 vertices
    False
    >>> graph = [{'a', 'b'}, {'b', 'c'}, {'c', 'd'}, {'d', 'e'}, {'e', 'a'}]
    >>> bipartite_graph_check(graph) # example for cycle graph with odd number of vertices
    False
    >>> graph = [{1, 2}, {2, 3}, {3, 1}, {1, 4}, {2, 4}, {3, 4}]
    >>> bipartite_graph_check(graph) # example for wheel graph
    False
    >>> graph = [{'01', '11'}, {'11', '10'}, {'10', '00'}]
    >>> bipartite_graph_check(graph) # example for hypercube graph
    True
    >>> graph = [('a', 'x'), ('a', 'y'), ('b', 'x')]
    >>> bipartite_graph_check(graph)
    True
    """
    if not graph:
        return "This graph is empty"

    def get_neighbouring_values(graph: list[tuple]) -> dict:
        """
        Function looks for every neighbouring vertix 
        and returns it as a dictionary.

        :param graph: list[tuple], given graph
        :return: dict, key is a vertex, value is all of the neighbouring vertices

        >>> graph = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (2, 1), (3, 1), (4, 2), (4, 3), (5, 4)]
        >>> get_neighbouring_values(graph)
        {1: {2, 3}, 2: {1, 4}, 3: {1, 4}, 4: {2, 3, 5}, 5: {4}}
        """
        neighbour_vertex = {}
        for v1, v2 in graph:
            if v1 not in neighbour_vertex:
                neighbour_vertex.setdefault(v1, set())
            if v2 not in neighbour_vertex:
                neighbour_vertex.setdefault(v2, set())

            neighbour_vertex[v1].add(v2)
            neighbour_vertex[v2].add(v1)
        return neighbour_vertex

    # getting values from additional functions
    neighbour_vertix = get_neighbouring_values(convert_to_directed(graph))

    # choosing a vertex to start with, setting its color
    start_vertix = list(neighbour_vertix)[0]
    color_vertices = {start_vertix: "green"}

    visited_vertices = set()
    not_visited_vertices = [start_vertix]

    while not_visited_vertices:
        # setting vertix to work with
        current_vertix = not_visited_vertices[0]
        if current_vertix not in visited_vertices:
            # visiting vertix
            visited_vertices.add(current_vertix)
            # setting color for neighbours
            color_vertix = "black" if color_vertices[current_vertix] == "green" else "green"
            for neighbour in neighbour_vertix[current_vertix]:
                if neighbour not in visited_vertices:
                    not_visited_vertices.append(neighbour)
                    color_vertices[neighbour] = color_vertix
                elif color_vertices[current_vertix] == color_vertices[neighbour]:
                    return False
        not_visited_vertices.remove(current_vertix)

    return True


def if_graphs_are_isomorphic(graph_1: list[tuple], graph_2: list[tuple]) -> bool:
    """
    Checks if two graphs are isomorphic

    :param graph_1: a list of edges for the first graph.
    :param graph_2: a list of edges for the second graph.
    :return: bool, True if the graphs are isomorphic, False if not.

    >>> if_graphs_are_isomorphic([(1, 2), (2, 1), (3, 4)], [(2, 1), (1, 2), (4, 3)])
    True
    >>> if_graphs_are_isomorphic([(1, 2), (2, 3)], [(3, 2), (2, 1)])
    True
    """

    def directed_isomorphism(graph_1: list[tuple], graph_2: list[tuple]) -> bool:
        """Checks if directed graphs are isomorphic
    
        :param graph_1: a list of directed edges for the first graph.
        :param graph_2: a list of directed edges for the second graph.
        :return: bool, True if the graphs are isomorphic, False if not.
    
         >>> directed_isomorphism([(1, 2), (2, 3), (3, 1)], [(3, 2), (2, 1), (1, 3)])
        True
        >>> directed_isomorphism([(1, 2), (2, 3), (3, 4)], [(4, 3), (3, 2), (2, 1)])
        False
        """
        if len(graph_1) != len(graph_2):
            return False

        vertices_1 = set()
        vertices_2 = set()

        for start, end in graph_1:
            vertices_1.add(start)
            vertices_1.add(end)
        for start, end in graph_2:
            vertices_2.add(start)
            vertices_2.add(end)

        if len(vertices_1) != len(vertices_2):
            return False

        for perm in permute(list(vertices_2)):
            vertex_match = {}
            for u, v in enumerate(vertices_1):
                vertex_match[v] = perm[u]

            new_graph = []
            for start, end in graph_1:
                new_graph.append((vertex_match[start], vertex_match[end]))

            if sorted(new_graph) == sorted(graph_2):
                return True

        return False


    def undirected_isomorphism(graph_1: list[set], graph_2: list[set]) -> bool:
        """
        Checks if undirected graphs are isomorphic
    
        :param graph_1: a list of undirected edges for the first graph.
        :param graph_2: a list of undirected edges for the second graph.
        :return: bool, True if the graphs are isomorphic, False if not.
    
        >>> undirected_isomorphism([{1, 2}, {2, 3}, {3, 4}], [{4, 3}, {2, 1}, {3, 2}])
        True
        >>> undirected_isomorphism([{1, 2}, {3, 4}], [{1, 2}, {3, 5}])
        False
        """
        if len(graph_1) != len(graph_2):
            return False

        vertices_1 = set()
        vertices_2 = set()

        for start, end in graph_1:
            vertices_1.add(start)
            vertices_1.add(end)
        for start, end in graph_2:
            vertices_2.add(start)
            vertices_2.add(end)

        if len(vertices_1) != len(vertices_2):
            return False

        for perm in permute(list(vertices_2)):
            vertex_match = {}
            for u, v in enumerate(vertices_1):
                vertex_match[v] = perm[u]

            new_graph = []
            for start, end in graph_1:
                new_graph.append((vertex_match[start], vertex_match[end]))
                new_graph.append((vertex_match[end], vertex_match[start]))

            if sorted(new_graph) == sorted(graph_2):
                return True

        return False

    if isinstance(graph_1[0], tuple) == isinstance(graph_2[0], tuple):
        if isinstance(graph_1[0], tuple) is True:
             return directed_isomorphism(graph_1, graph_2)
        else:
            return undirected_isomorphism(graph_1, graph_2)
    else:
        return False


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
