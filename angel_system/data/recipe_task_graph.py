import os 
import yaml
import random
import argparse

import networkx as nx
import matplotlib.pyplot as plt


dir_path = os.path.dirname(os.path.realpath(__file__))
def load_recipe(recipe):
    config_dir = f"{dir_path}/../../config/activity_labels"
    
    if recipe == "tea":
        config_fname = f"{config_dir}/recipe_tea.yaml"
    
    with open(config_fname, "r") as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded recipe config: {config_fname}")

    recipe_title = config["title"]

    labels = config["labels"]
    labels = labels[1:] # Ignore background
    
    # Add start
    labels.append(
        {
            "id": 0,
            "label": "start",
            "full_str": "start",
            "depends": None
        }
    )

    return recipe_title, labels

def get_all_recipe_orders(labels):
    G = nx.DiGraph()

    for action in labels:
        act_label = action["label"]
        G.add_node(act_label)
        
        depends = action["depends"]
        if depends is not None:
            for d in depends:
                G.add_edge(d, act_label)

    all_possible_orders = list(nx.all_topological_sorts(G))
    for t in all_possible_orders:
        # Some sanity checks
        # Check sub step order
        assert t.index("pour-water-kettle") > t.index("measure-12oz-water")
        assert t.index("stir") > t.index("add-honey")
        assert t.index("check-thermometer") > t.index("thermometer-turn-on")
        assert t.index("thermometer-in-water") > t.index("thermometer-turn-on")
        assert t.index("check-thermometer") > t.index("thermometer-in-water")
        assert t.index("stir") > t.index("add-honey")
        assert t.index("discard-tea-bag") > t.index("remove-tea-bag")

        # Check stop points order
        stop_point1 = ["measure-12oz-water", "pour-water-kettle", "tea-bag-in-mug", "thermometer-turn-on", "thermometer-in-water", "check-thermometer", "pour-water-mug"]
        stop_point2 = ["steep", "remove-tea-bag", "discard-tea-bag", "add-honey", "stir"]
        for n in stop_point1:
            assert t.index("steep") > t.index(n)
        
    # print(len(all_possible_orders))
    return all_possible_orders

def print_recipe_order(recipe_title, labels, all_possible_orders):
    # nx.draw(G, with_labels=True)
    # plt.savefig("tea_graph.png")
    # TODO: create sample bias towards regular order?
    recipe_order = random.choice(all_possible_orders)

    print(f"\n{recipe_title}")
    print("===============")
    for i, action_label in enumerate(recipe_order):
        action = [l for l in labels if l["label"] == action_label][0]
        str_ = action["full_str"]
        if str_ in ["start", "done"]:
            continue
        print(f"{i}: {str_}")

    print("\n")
    print(f"Recipe order: {recipe_order}")
    print("* Please copy and paste this into the 'recipe order' column of the 'PTG Cooking Data Collection' spreadsheet")

def main():
    all_recipes = ['coffee', 'tea', 'oatmeal', 'pinwheel', 'dessertquesadilla']
    
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
