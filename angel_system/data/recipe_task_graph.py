import os
import yaml
import random
import warnings
import argparse

import networkx as nx
import matplotlib.pyplot as plt

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
    labels.append({"id": 0, "label": "start", "full_str": "start", "depends": None})

    return recipe_title, labels


def get_all_recipe_orders(labels):
    """
    Creates a recipe graph and determines all
    possible topological sortings of the graph

    :param labels: List of activity labels

    :return: A tuple consisting of two elements:
        0: List of all possible topological sortings of the recipe graph
        1: List of ids that represent the root node of a set of repeated actions
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

    all_possible_orders = list(nx.all_topological_sorts(G))
    return all_possible_orders, repeated_ids


def print_recipe_order(recipe_title, labels, all_possible_orders, repeated_ids):
    """
    Prints a list of instructions for a recipe

    :param recipe_title: String containing the full title of the recipe
    :param labels: List of activity labels
    :param all_possible_orders: List of all possible topological sortings of the recipe graph
    :param repeated_ids: List of ids that represent the root node of a set of repeated actions.
        These ids are skipped when printing the recipe steps to avoid duplication of instructions
    """
    # TODO: create sample bias towards regular order?
    recipe_order = random.choice(all_possible_orders)

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

    all_possible_orders, repeated_ids = get_all_recipe_orders(labels)
    print_recipe_order(recipe_title, labels, all_possible_orders, repeated_ids)


if __name__ == "__main__":
    main()
