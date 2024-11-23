#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
from pathlib import Path
import random
import statistics
import textwrap
from random import randint
from typing import Iterator, Dict, List
import matplotlib
import matplotlib.pyplot as plt
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
)

matplotlib.use("Agg")

random.seed(9001)

__author__ = "Jessica Buszta"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Jessica Buszta"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Jessica Buszta"
__email__ = "jessica.buszta@etu.u-paris.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file,'rt') as monfichier:
        for line in monfichier:
            yield next(monfichier).replace('\n','')
            next(monfichier)
            next(monfichier)


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(0,len(read)-kmer_size+1):
        yield read[i:i+kmer_size]


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    d = {}
    for seq in read_fastq(fastq_file):
        for kmer in cut_kmer(seq,kmer_size):
            if kmer in d:
                d[kmer]+=1
            else:
                d[kmer]=1
    return d


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    digraph = DiGraph()
    for kmer in kmer_dict:
        digraph.add_edge(kmer[:-1], kmer[1:], weight=kmer_dict[kmer])
    return digraph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        if  delete_entry_node and delete_sink_node:
            graph.remove_nodes_from(path)
        elif delete_entry_node and not delete_sink_node:
            graph.remove_nodes_from(path[:-1])
        elif not delete_entry_node and delete_sink_node:
            graph.remove_nodes_from(path[1:])
        else:
            graph.remove_nodes_from(path[1:-1])
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    if statistics.stdev(weight_avg_list) > 0:
        best_path_index = weight_avg_list.index(max(weight_avg_list))
    elif statistics.stdev(path_length) > 0:
        best_path_index = path_length.index(max(path_length))
    else:
        best_path_index = randint(0,len(path_list) - 1)

    del path_list[best_path_index]
    graph = remove_paths(graph, path_list, delete_entry_node, delete_sink_node)
    return graph


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    path_list = list(all_simple_paths(graph, ancestor_node, descendant_node))
    path_length = []
    weight_avg_list = []
    for path in path_list:
        path_length += [len(path)]
        weight_avg_list += [path_average_weight(graph, path)]
    return select_best_path(graph, path_list, path_length, weight_avg_list)


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble = False
    for node in graph:
        list_pred = list(graph.predecessors(node))
        if len(list_pred) > 1:
            # On teste les combinaisons uniques de (i, j) où i < j
            for i in range(len(list_pred) - 1):
                for j in range(i+1, len(list_pred)):
                    anc_node = lowest_common_ancestor(graph, list_pred[i], list_pred[j])
                    if anc_node:
                        bubble = True
                        break
    if bubble:
        graph = simplify_bubbles(solve_bubble(graph, anc_node, node))

    return graph


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    list_path = []
    list_weight = []
    path_length = []

    for node in graph:
        if node not in starting_nodes and len(list(graph.predecessors(node))) > 1:
            for starting_node in starting_nodes:
                if has_path(graph, starting_node, node):
                    for path in all_simple_paths(graph, starting_node, node):
                        list_path.append(path)
                        list_weight.append(path_average_weight(graph, path))
                        path_length.append(len(path))
            break
    if len(list_path) > 1:
        graph = select_best_path(
            graph,
            path_list=list_path,
            weight_avg_list=list_weight,
            path_length=path_length,
            delete_entry_node=True,
            delete_sink_node=False,
        )
        return solve_entry_tips(graph, starting_nodes)

    return graph


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    list_path = []
    list_weight = []
    path_length = []
    for node in graph:
        if node not in ending_nodes and len(list(graph.successors(node))) > 1:
            for ending_node in ending_nodes:
                if has_path(graph, node, ending_node):
                    for path in all_simple_paths(graph, node, ending_node):
                        list_path.append(path)
                        list_weight.append(path_average_weight(graph, path))
                        path_length.append(len(path))
            break
    if len(list_path) > 1:
        graph = solve_out_tips(
            select_best_path(
                graph,
                path_list=list_path,
                weight_avg_list=list_weight,
                path_length=path_length,
                delete_entry_node=False,
                delete_sink_node=True,
            ),
            ending_nodes,
        )

    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    nodelist = []
    for node in graph:
        if len(list(graph.predecessors(node))) == 0:
            nodelist += [node]
    return nodelist


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    nodelist = []
    for node in graph:
        if len(list(graph.successors(node))) == 0:
            nodelist += [node]
    return nodelist


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigslist = []
    for starting_node in starting_nodes:
        for ending_node in ending_nodes:
            if has_path(graph, starting_node, ending_node):
                for path in all_simple_paths(graph, starting_node, ending_node):
                    contig = path[0]
                    for node in path[1:]:
                        contig += node[-1]
                    contigslist += [[contig, len(contig)]]
    return contigslist

def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file,"w") as f:
        for i, contig in enumerate(contigs_list):
            f.write(f">contig_{i} len={contig[1]}\n")
            f.write(textwrap.fill(contig[0],width=80))
            f.write("\n")

def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = random_layout(graph)
    draw_networkx_nodes(graph, pos, node_size=6)
    draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Build Graph
    kmer_dict = build_kmer_dict(args.fastq_file, args.kmer_size)
    graph = build_graph(kmer_dict)

    # Bubbles solving
    graph = simplify_bubbles(graph)

    # Tips solving
    starting_nodes = get_starting_nodes(graph)
    sink_nodes = get_sink_nodes(graph)

    graph = solve_entry_tips(graph, starting_nodes)
    graph = solve_out_tips(graph, sink_nodes)

    # Get contigs
    contigs_list = get_contigs(graph, starting_nodes, sink_nodes)

    # Save contigs
    save_contigs(contigs_list, args.output_file)

    # Plot the graph
    if args.graphimg_file:
        draw_graph(graph, args.graphimg_file)


if __name__ == "__main__":  # pragma: no cover
    main()
