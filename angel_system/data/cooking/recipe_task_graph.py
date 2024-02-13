import argparse
import os
import random
from typing import Dict
from typing import List
from typing import Optional
import warnings
import yaml

import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from pathlib import Path


dir_path = Path(__file__).parent


def load_recipe(recipe):
    """
    Loads a recipe action configuration file

    :param recipe: String representing the name of the recipe to run

    :return: A tuple consisting of two elements:
        0: String containing the full title of the recipe
        1: List of activity labels
    """
    config_dir = f"{dir_path}/../../config/activity_labels"
    assert os.path.isdir(
        config_dir
    ), f"Configuration directory {config_dir} does not exist"

    config_fname = f"{config_dir}/recipe_{recipe}.yaml"
    assert os.path.exists(config_fname), f"Configuration {config_fname} does not exist"

    with open(config_fname, "r") as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded recipe config: {config_fname}")

    recipe_title = config["title"]

    labels = config["labels"]
    labels = labels[1:]  # Ignore background

    # Add start
    labels.insert(0, {"id": 0, "label": "start", "full_str": "start", "depends": None})

    return recipe_title, labels


def create_graph_from_labels(labels: List[Dict]):
    """
    Creates a NetworkX Graph instance of the recipe graph given as a list of
    label dicts.

    See `load_recipe` for dict format expected.

    :param labels: List of activity labels dictionaries that specify the graph.

    :return: NetworkX Graph instance constructed, as well as a list of label
        IDs that represent the root node of a set of repeated actions.
    """
    G = nx.DiGraph()
    repeated_ids = []

    for action in labels:
        act_id = action["id"]
        G.add_node(act_id)

        if act_id == 0:
            continue

        depends = action.get("depends", None)

        if "repeat" in action.keys():
            repeat = action["repeat"]
            repeated_ids.append(act_id)

            for r in range(repeat):
                new_act_id = f"{act_id}-{r}"
                G.add_edge(new_act_id, act_id)

                G.add_node(new_act_id, root_label=act_id)
                if depends is not None:
                    for d in depends:
                        G.add_edge(d, new_act_id)
                else:
                    G.add_edge(0, new_act_id)
        else:
            if depends is not None:
                for d in depends:
                    G.add_edge(d, act_id)
            else:
                G.add_edge(0, act_id)

    return G, repeated_ids


def get_one_recipe_order(g: nx.DiGraph, seed: Optional[int] = None) -> List:
    """
    Create a recipe graph from the input labels and determine a single possible
    topological sorting of the graph.

    :param g: Recipe graph as would be output from `create_graph_from_labels`.
    :param seed: Optional random state seed integer. None means no seed will be
        used.

    :return: A list of node IDs in a random topological sorting as influenced
        by the given seed.
    """
    g_prime = g.copy()
    rng = random.Random(seed)
    topo_sort = []
    while len(g_prime) > 0:
        zero_in_degree = [v for v, d in g_prime.in_degree() if d == 0]
        rand_zid_node = rng.choice(zero_in_degree)
        g_prime.remove_node(rand_zid_node)
        topo_sort.append(rand_zid_node)
    return topo_sort


def get_all_recipe_orders(labels):
    """
    Creates a recipe graph and determines all
    possible topological sortings of the graph

    :param labels: List of activity labels

    :return: A tuple consisting of two elements:
        0: List of all possible topological sortings of the recipe graph
        1: List of ids that represent the root node of a set of repeated actions
    """
    G, repeated_ids = create_graph_from_labels(labels)
    all_possible_orders = list(
        tqdm(
            nx.all_topological_sorts(G),
            desc="Iterating all topological orders",
            unit=" orders",
        )
    )
    return all_possible_orders, repeated_ids


def print_recipe_order(recipe_title, labels, recipe_order, repeated_ids):
    """
    Prints a list of instructions for a recipe

    :param recipe_title: String containing the full title of the recipe
    :param labels: List of activity labels
    :param recipe_order: List of IDs specifying one possible topological sorting of the recipe graph.
    :param repeated_ids: List of ids that represent the root node of a set of repeated actions.
        These ids are skipped when printing the recipe steps to avoid duplication of instructions
    """
    if recipe_title == "Dessert Quesadilla":
        # Remove extra clean
        clean_1 = recipe_order.index(10)
        clean_2 = recipe_order.index(11)

        recipe_order.pop(min(clean_1, clean_2))

    print(f"\n{recipe_title}")
    print("=" * len(recipe_title))
    recipe_actions = [
        al for al in recipe_order if al not in [0, len(labels) - 1] + repeated_ids
    ]
    for i, action_id in enumerate(recipe_actions):
        if type(action_id) is int:
            action = [l for l in labels if l["id"] == action_id][0]
        elif "-" in str(action_id):
            # The label might have a repeat index attached to it
            action = [l for l in labels if l["id"] == int(action_id.split("-")[0])][0]
        else:
            warnings.warn(f"Unknown action id: {action_id}")
            continue

        str_ = action["full_str"]

        print(f"{i+1}: {str_}")

    print("\n")
    print(f"Recipe order: {recipe_order}")
    print(
        "* Please copy and paste this into the 'recipe order' column of the 'PTG Cooking Data Collection' spreadsheet"
    )


def main():
    all_recipes = ["coffee", "tea", "oatmeal", "pinwheel", "dessertquesadilla"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe",
        type=str,
        default="tea",
        help=f"Title of recipe to run. Options are {all_recipes}",
    )

    args = parser.parse_args()
    if args.recipe not in all_recipes:
        print(f"Must select one of: {all_recipes}")
        return

    recipe_title, labels = load_recipe(args.recipe)

    # -----------------------------------------------------------------------
    # New method of just creating a single random topological sorting
    G, repeated_ids = create_graph_from_labels(labels)
    final_order = None
    attempts = 0
    while final_order is None:
        # TODO: create sample bias towards regular order?
        order = get_one_recipe_order(G)
        attempts += 1

        # Special case for dessert quesadillas -- check for knife cleaning
        # complexity.
        # TODO: determine a better way of doing this...
        if args.recipe == "dessertquesadilla":
            clean_knife_n = order.index(11)
            scoop_nutella = order.index(2)

            clean_knife_b = order.index(10)
            slice_banana = order.index(4)

            spread_nutella = order.index(3)

            if (
                (scoop_nutella == clean_knife_n + 1)
                and (slice_banana == clean_knife_b + 1)
                and (spread_nutella == scoop_nutella + 1)
            ):
                print(
                    f"Made {attempts} attempts before arriving at a valid "
                    f"`dessertquesadilla` order."
                )
                final_order = order
            # Otherwise, keep generating random orders until we find one with a
            # valid happen-stance ordering.
        else:
            final_order = order

    print_recipe_order(recipe_title, labels, final_order, repeated_ids)

    # # -----------------------------------------------------------------------
    # # Functional version of the previous, overly-exhaustive method.
    # all_possible_orders, repeated_ids = get_all_recipe_orders(labels)
    #
    # if args.recipe == "dessertquesadilla":
    #     # Make sure the knife is cleaned directly before using it
    #     cleaned_possible_orders = []
    #     for order in all_possible_orders:
    #         clean_knife_n = order.index(11)
    #         scoop_nutella = order.index(2)
    #
    #         clean_knife_b = order.index(10)
    #         slice_banana = order.index(4)
    #
    #         spread_nutella = order.index(3)
    #
    #         if (
    #             (scoop_nutella == clean_knife_n + 1)
    #             and (slice_banana == clean_knife_b + 1)
    #             and (spread_nutella == scoop_nutella + 1)
    #         ):
    #             cleaned_possible_orders.append(order)
    #     all_possible_orders = cleaned_possible_orders
    #
    # final_order = random.choice(all_possible_orders)
    #
    # print_recipe_order(recipe_title, labels, final_order, repeated_ids)


if __name__ == "__main__":
    main()
