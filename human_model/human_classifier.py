def human_classify(wine_obj):
    if wine_obj["density"] < 0.99007:
        return "white"
    elif wine_obj["fixed_acidity"] > 12:
        return "red"
    elif wine_obj["density"] > 9912:
        return "white"
    elif wine_obj["chlorides"] < 0.046:
        return "white"
    elif wine_obj["sulphates"] > 1:
        return "red"
    elif wine_obj["free_sulfur_dioxide"] > 75:
        return "white"
    elif wine_obj["total_sulfur_dioxide"] > 160:
        return "white"
    elif wine_obj["residual_sugar"] < 10:
        return "red"

    else:
        return "unknown"