import os
import yaml
import random
import argparse

import networkx as nx
import matplotlib.pyplot as plt


dir_path = os.path.dirname(os.path.realpath(__file__))


def load_recipe(recipe):
    config_dir = f"{dir_path}/../../config/activity_labels"
    config_fname = f"{config_dir}/recipe_{recipe}.yaml"

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
    G = nx.DiGraph()

    for action in labels:
        act_label = action["label"]
        depends = action["depends"]

        if "repeat" in action.keys():
            repeat = action["repeat"]

            # Node that indicated when the full repetition is finished
            all_act_label = f"{act_label}-all"
            G.add_node(all_act_label)
            for r in range(repeat):
                new_act_label = f"{act_label}-{r}"
                G.add_edge(new_act_label, all_act_label)

                G.add_node(new_act_label, root_label=act_label)
                if depends is not None:
                    for d in depends:
                        G.add_edge(d, new_act_label)
        else:
            G.add_node(act_label)

            if depends is not None:
                for d in depends:
                    G.add_edge(d, act_label)

    # nx.draw(G, with_labels=True)
    # plt.savefig("oatmeal_graph.png")

    all_possible_orders = list(nx.all_topological_sorts(G))
    return all_possible_orders


def print_recipe_order(recipe_title, labels, all_possible_orders):
    # TODO: create sample bias towards regular order?
    recipe_order = random.choice(all_possible_orders)

    print(f"\n{recipe_title}")
    print("=" * len(recipe_title))
            
    recipe_actions = [al for al in recipe_order if not al.endswith("-all")]
    for i, action_label in enumerate(recipe_actions):
        try:
            action = [l for l in labels if l["label"] == action_label][0]
        except:
            # The label might have a repeat index attached to it
            action = [l for l in labels if l["label"] == '-'.join(action_label.split('-')[:-1])][0]
        str_ = action["full_str"]
        if str_ in ["start", "done"]:
            continue
        print(f"{i}: {str_}")

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

    all_possible_orders = get_all_recipe_orders(labels)
    print_recipe_order(recipe_title, labels, all_possible_orders)


if __name__ == "__main__":
    main()
