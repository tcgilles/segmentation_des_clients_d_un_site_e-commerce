import pandas as pd
import numpy as np
import random
from random import choices
from math import radians, cos, sin, asin, sqrt
from time import time


def find_subtring(string, subtrings):
    """
    Detects if at least one of the substrings is in the string
    Args:
        string (String): the text in which we are looking for the words
        subtrings (List): list of the substrings that we want to check 
                          the occurence of in the string
    Returns:
        True if there is at least one word in words that's in text, 
        False otherwise
    """
    # For each subtring in subtrings,
    for subtring in subtrings :
        
        # if subtring in string, return True and exit the function
        if string.find(subtring) != -1 : 
            return True
        
    # otherwise, return false
    return False


def fill_mean(dfunc, feature_cat, feature_num):
    """
    Pairs each row in feature_cat with the mean value of the category 
    associated with that row. The values are those of feature_num.
    Args:
        dfunc (DataFrame)
        feature_cat (string) : a single categorical column of dfunc
        feature_num (string) :  a single numerical column of dfunc
    Returns:
        categ_means (pd.Series)  
    """
    # groupby dfunc with feature_cat
    dfunc_gb = dfunc.groupby(feature_cat)[feature_num].mean().reset_index()
    
    # means : keys are the categories in feature_cat, 
    #         values are the mean values of each category 
    means={}
    
    for category in dfunc_gb[feature_cat].unique():
        means[category] = dfunc_gb.loc[dfunc_gb[feature_cat]==category, 
                                       feature_num].values[0]
     
    categ_means = pd.Series(dfunc[feature_cat].map(means).values)
    
    return categ_means


def distance(lat1, lon1, lat2, lon2):
    '''
    Haversine distance between 2 points
    Args :
        lat1, lon1 (float): latitude and longitude of the first point
        lat2, lon2 (float): latitude and longitude of the second point
    Returns : 
        d (float): the haversine distance between the 2 points
    '''
    # Conversion en radians
    lon1, lon2, lat1, lat2 = map(radians, [lon1, lon2, lat1, lat2])
      
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers.
    r = 6371
      
    # calculate the result
    return c * r


def clean_orders():
    '''
    achieves all the cleaning and feature engineering of the P5 project
    Args : None
    Returns : 
        orders : the dataframe of all the orders cleaned
        customers : the dataframe of customers orders summary
    '''
    start_time = time()
    
    print("Ouverture des fichiers...")
    
    # Lecture des fichiers
    customers = pd.read_csv("olist_customers_dataset.csv")
    geolocation = pd.read_csv("olist_geolocation_dataset.csv")
    order_items = pd.read_csv("olist_order_items_dataset.csv")
    order_payments = pd.read_csv("olist_order_payments_dataset.csv")
    order_reviews = pd.read_csv("olist_order_reviews_dataset.csv")
    orders = pd.read_csv("olist_orders_dataset.csv")
    products = pd.read_csv("olist_products_dataset.csv")
    translation = pd.read_csv("product_category_name_translation.csv")
    
    print("Nettoyage des dataset individuels...")
    
    # On supprime les commandes pas encore livrées
    mask = orders["order_status"] == "delivered"
    orders = orders.loc[mask, :]
    
    # Pour les commandes qui ont été livrées et dont la date de livraison 
    # au client est inconnu, nous supposons que la livraison a eu lieu à la
    # date de livraison estimée lors de l'achat
    mask1 = orders["order_delivered_customer_date"].isna()
    mask2 = orders["order_status"] == "delivered"
    mask = (mask1) & (mask2)
    orders.loc[mask, "order_delivered_customer_date"] = \
        orders.loc[mask, "order_estimated_delivery_date"]
    
    # On supprime certaines colonnes
    cols_to_delete = ["order_status", "order_approved_at", 
                      "order_delivered_carrier_date", 
                      "order_estimated_delivery_date",]
    orders = orders.drop(columns=cols_to_delete)
    
    # Nous allons corriger le type de la colonne order_purchase_timestamp et 
    # order_delivered_customer_date
    orders["order_purchase_timestamp"] = \
        pd.to_datetime(orders["order_purchase_timestamp"], 
                    format="%Y-%m-%d %H:%M:%S")

    orders["order_delivered_customer_date"] = \
        pd.to_datetime(orders["order_delivered_customer_date"], 
                    format="%Y-%m-%d %H:%M:%S")
    
    # On supprime certaines colonnes
    cols_to_delete = ["customer_city",]
    customers = customers.drop(columns=cols_to_delete)
    
    cols_to_delete = ["shipping_limit_date", "seller_id",]
    order_items = order_items.drop(columns=cols_to_delete)
    
    # Nous allons faire un groupby de sorte à avoir des couples uniques 
    # (order_id, product_id)
    order_items = order_items.groupby(["order_id", "product_id"])\
                             .agg({"order_item_id": np.max,
                               "price": np.sum,
                               "freight_value": np.sum}).reset_index()
    
    # Nous allons travailler avec les longitudes et latitudes moyennes des 
    # différents code_postaux, de sorte à avoir des clés uniques dans la 
    # colonne geolocation_zip_code_prefix
    geolocation = geolocation.groupby("geolocation_zip_code_prefix")\
                             .agg({"geolocation_lat" : np.mean,
                                   "geolocation_lng" : np.mean}).reset_index()
    
    # On supprime des colonnes
    cols_to_delete = ["payment_type",]
    order_payments = order_payments.drop(columns=cols_to_delete)
    
    # Nous allons effectuer un groupby pour n'avoir que des clés uniques 
    # dans la colonne order_id
    order_payments = order_payments.groupby("order_id")\
                                   .agg({"payment_sequential": np.max, 
                                     "payment_installments": np.sum, 
                                     "payment_value": np.sum,})\
                                   .reset_index()
    
    # Renommons les colonnes
    order_payments.rename(columns={"payment_sequential": "nb_payment_method", 
                               "payment_installments": "nb_installments"}, 
                          inplace=True)
    
    # Nous allons corriger le type de la colonne review_creation_date
    order_reviews["review_answer_timestamp"] = \
        pd.to_datetime(order_reviews["review_answer_timestamp"], 
                       format="%Y-%m-%d %H:%M:%S")
    
    # Nous ne conserverons que l'évaluation la plus récente
    order_reviews = order_reviews.sort_values(["review_creation_date"], 
                                              ascending=False)
    order_reviews = order_reviews.drop_duplicates(subset=["order_id"])\
                                 .sort_index()
    
    # Colonnes à supprimer
    cols_to_delete = ["review_id", "review_comment_title", 
                      "review_comment_message", "review_creation_date", 
                      "review_answer_timestamp",]
    order_reviews = order_reviews.drop(columns=cols_to_delete)
    
    # Nous ne conserverons que les colonnes product_id et product_category_name
    products = products.loc[:, ["product_id", "product_category_name"]]
    
    # Nous allons attribuer la valeur unknown aux articles dont la 
    # catégorie n'est pas renseignée.
    products["product_category_name"] = \
        products["product_category_name"].fillna("unknown")
    
    # Traduction des catégories en anglais
    product_categories_eng = dict(translation.values)
    products["product_category_eng"] = \
        products["product_category_name"].map(product_categories_eng)

    # Certaines des catégories en protugais n'ont pas été traduites en anglais. 
    # Du coup, des NaN sont apparues suite à l'opération ci-dessus.
    products["product_category_name"] = \
        products["product_category_eng"].fillna(products["product_category_name"])

    # Nous allons maintenant supprimer la colonne "product_category_eng"
    products = products.drop(columns="product_category_eng")
    
    # Regroupons les catégories en suivant le modèle d'AMAZON
        # Mots clés des nouvelles catégories
    loisirs = ["leisure", "consoles", "audio", "dvds", "music", "books", "art"]
    hightech = ["telephony", "stuff", "electronics", "instruments", "photo", 
                 "image", "gamer", "services"]
    informatique_bureau = ["computers", "stationery", "office"]
    jouets_bébé = ["toys", "baby", "diapers"]
    maison_déco = ["table", "furniture", "house", "appliances", "conditioning", 
                   "supplies", "flowers", "cuisine", "home"]
    brico_jardin_animaux = ["garden", "pet", "tools", "signaling"]
    beauté_santé = ["health", "perfume"]
    aliments_autres = ["food", "drinks", "portateis", "unknown"]
    fashion_accessoires = ["watches", "accessories", "fashio"]
    auto_industrie = ["auto", "industry", "market"]

        # On enregistre dans une liste
    new_categories = {"hightech": hightech, 
                      "maison_déco": maison_déco, 
                      "loisirs": loisirs, 
                      "informatique_bureau": informatique_bureau, 
                      "jouets_bébé": jouets_bébé, 
                      "brico_jardin_animaux": brico_jardin_animaux, 
                      "beauté_santé": beauté_santé, 
                      "aliments_autres": aliments_autres, 
                      "fashion_accessoires": fashion_accessoires, 
                      "auto_industrie": auto_industrie}

        # On effectue le regroupement des catégories
    for categ in new_categories:
        keywords = new_categories[categ]

            # On sélectionne les lignes dont la catégorie contient au moins
            # un des mots clés
        mask = products["product_category_name"].apply(find_subtring, 
                                                       args=([keywords]))

            # On modifie le nom de la catégorie
        products.loc[mask, "product_category_name"] = categ
    
    print("Merge des dataset pour reconstitution du dataset des commandes...")
    
    # Merge entre customers et geolocation
    geolocation.rename(columns={"geolocation_zip_code_prefix": \
                                "customer_zip_code_prefix"}, 
                       inplace=True)
    customers = pd.merge(customers, geolocation, on="customer_zip_code_prefix", 
                         how="left")
    
        # On va supprimer la colonne customer_zip_code_prefix
    customers = customers.drop(columns=["customer_zip_code_prefix"])
    
        # Imputons les valeurs manquantes de geolocation_lat et geolocation_lng
    customers["geolocation_lat"] = \
        customers["geolocation_lat"].fillna(fill_mean(customers, 
                                                      "customer_state", 
                                                      "geolocation_lat"))
    customers["geolocation_lng"] = \
        customers["geolocation_lng"].fillna(fill_mean(customers, 
                                                      "customer_state", 
                                                      "geolocation_lng"))
    
    # Merge entre products et order_items
    order_items = pd.merge(order_items, products, on="product_id", how="left")
    
    # Nous allons maintenant créer 10 variables synthétiques qui représenteront 
    # le montant dépensé dans chaque catégorie d'article par commande.
        # On enregistre les catégories des articles dans une liste
    categories = order_items.product_category_name.unique().tolist()

        # Pour chaque catégorie d'article
    for categ in categories:

            # Nom de la nouvelle feature
        feature = f"{categ}" + "_value"

            # On initialise la valeur dépensée sur cette catégorie
        order_items[feature] = 0

            # On calcule le montant dépensé dans cette catégorie pour chaque 
            # commande
        mask = order_items["product_category_name"] == categ
        order_items.loc[mask, feature] = order_items.loc[mask, "price"]
        
    # On va maintenant faire un groupby afin de conserver des clés uniques 
    # dans la colonne order_id
    order_items = order_items.groupby(["order_id"])\
                         .agg({"order_item_id": np.max, 
                               "price": np.sum, 
                               "freight_value": np.sum, 
                               "hightech_value": np.sum,
                               "brico_jardin_animaux_value": np.sum,
                               "maison_déco_value": np.sum,
                               "beauté_santé_value": np.sum,
                               "loisirs_value": np.sum,
                               "fashion_accessoires_value": np.sum,
                               "aliments_autres_value": np.sum,
                               "auto_industrie_value": np.sum,
                               "informatique_bureau_value": np.sum,
                               "jouets_bébé_value": np.sum,})\
                         .reset_index()
    
        # Renommons les colonnes
    order_items.rename(columns={"order_item_id": "nb_items",}, inplace=True)
    
    # Merge entre customers et orders
    orders = pd.merge(orders, customers, on="customer_id", how="left")
    
    # Merge entre reviews et orders
    orders = pd.merge(orders, order_reviews, on="order_id", how="left")
    
    # Merge entre order_payments et orders
    orders = pd.merge(orders, order_payments, on="order_id", how="left")
    
    # Merge entre order_items et orders
    orders = pd.merge(orders, order_items, on="order_id", how="left")
    
    # Nous allons maintenant supprimer la colonne customer_id
    orders = orders.drop(columns=["customer_id"])
    
    # Imputons les valeurs manquantes des colonnes nb_payment_method, 
    # nb_installments et payment_value
    orders["nb_payment_method"] = \
    orders["nb_payment_method"].fillna(orders["nb_payment_method"].mode().values[0])

    orders["nb_installments"] = \
    orders["nb_installments"].fillna(orders["nb_installments"].mode().values[0])

    orders["payment_value"] = \
    orders["payment_value"].fillna(orders["price"]+orders["freight_value"])

    # Certaines des valeurs de nb_installments sont nulles, ce qui est 
    # impossible. Nous les attribuerons la valeur de 1.
    mask = orders["nb_installments"] == 0
    orders.loc[mask, "nb_installments"] = 1
    
    # Nous allons remplacer les valeurs de payment_value par 
    # price+freight_value et suppriemr price
    orders["payment_value"] = orders["price"] + orders["freight_value"]
    orders = orders.drop(columns=["price"])
    
    # imputons les valeurs manquantes de review_score
    random.seed(42)
    population = orders["review_score"].value_counts().index.tolist()
    weights = orders["review_score"].value_counts().values.tolist()
    missing_reviews = choices(population, weights, 
                              k=orders["review_score"].isna().sum(),)

    mask = orders["review_score"].isna()
    orders.loc[mask, "review_score"] = missing_reviews
    
    # Nous allons mettre les numéros de commande en index
    orders.set_index("order_id", inplace=True)
    
    # Nous allons trier le dataset par date d'achat
    orders.sort_values("order_purchase_timestamp", inplace=True)
    
    print("Début du feature engineering...")
    
    # Coordonnées des locaux de Olist
    olist_lat = -25.43043943931033
    olist_lng = -49.292139311888576

    # Calcul de la distance entre le client et les locaux de Olist
    orders["distance"] = [distance(lat, lng, olist_lat, olist_lng) \
                          for lat,lng in zip(orders["geolocation_lat"], 
                                             orders["geolocation_lng"])]

    # Suppression des colonnes
    orders.drop(columns=["geolocation_lat", "geolocation_lng", 
                         "customer_state"], inplace=True)
    
    # Délai de livraison
    orders["delivery_time"] = \
        (orders["order_delivered_customer_date"] - \
         orders["order_purchase_timestamp"]) / np.timedelta64(1,"D")

    # Suppression des colonnes
    orders.drop(columns=["order_delivered_customer_date"], inplace=True)
    
    # Mois d'achat
    orders["month_of_purchase"] = orders["order_purchase_timestamp"].dt.month
    
    print("Création du fichier client...")
    # Date la plus récente dans le dataset
    last_date = orders["order_purchase_timestamp"].max()

    # Création du fichier client
    customers = orders.groupby(["customer_unique_id"]).agg(
        nb_orders=pd.NamedAgg(column="customer_unique_id", aggfunc="count"),
        distance=pd.NamedAgg(column="distance", aggfunc="last"),
        delivery_time=pd.NamedAgg(column="delivery_time", aggfunc="mean"),
        fav_month_of_purchase=pd.NamedAgg(column="month_of_purchase", 
                                          aggfunc=lambda x: x.mode()[0]),
        most_recent_purchase=pd.NamedAgg(column="order_purchase_timestamp", 
                aggfunc=lambda x: (last_date-x.max())/np.timedelta64(1,"D")),
        review_score=pd.NamedAgg(column="review_score", aggfunc="mean"),
        nb_payment_method=pd.NamedAgg(column="nb_payment_method", aggfunc="mean"),
        nb_installments=pd.NamedAgg(column="nb_installments", aggfunc="mean"),
        nb_items=pd.NamedAgg(column="nb_items", aggfunc="sum"),
        amount_spent=pd.NamedAgg(column="payment_value", aggfunc="sum"),
        freight_ratio=pd.NamedAgg(column="freight_value", aggfunc="sum"),
        hightech_spent=pd.NamedAgg(column="hightech_value", aggfunc="sum"),
        brico_jardin_animaux_spent=\
                pd.NamedAgg(column="brico_jardin_animaux_value", aggfunc="sum"),
        maison_déco_spent=pd.NamedAgg(column="maison_déco_value", aggfunc="sum"),
        beauté_santé_spent=pd.NamedAgg(column="beauté_santé_value", aggfunc="sum"),
        loisirs_spent=pd.NamedAgg(column="loisirs_value", aggfunc="sum"),
        fashion_accessoires_spent=\
                pd.NamedAgg(column="fashion_accessoires_value", aggfunc="sum"),
        aliments_autres_spent=pd.NamedAgg(column="aliments_autres_value", 
                                          aggfunc="sum"),
        auto_industrie_spent=pd.NamedAgg(column="auto_industrie_value", 
                                         aggfunc="sum"),
        informatique_bureau_spent=\
                pd.NamedAgg(column="informatique_bureau_value", aggfunc="sum"),
        jouets_bébé_spent=pd.NamedAgg(column="jouets_bébé_value", aggfunc="sum"),
    )
    customers["freight_ratio"] = customers["freight_ratio"]/customers["amount_spent"]

    end_time = time() - start_time
    print("Fin du traitement")
    print(f"Temps écoule : {end_time :.1f}s")
    
    return orders, customers


def get_customers(orders, customers, start=0, stop=15):
    '''
    create the customer dataset from the orders dataset
    Args :
        orders (pd.DataFrame): the dataset of all the orders cleaned
        customers (pd.DataFrame): rhe dataset of customers
        start, stop (floats, unit=days): select only the orders between
                                 [oldest date + start, oldest date + stop]
    Returns : 
        the Dataframe of customers
    '''
    start_time = time()
    print("Création du fichier client...")

    # Sélection des clients entre start et stop
    start_date = orders["order_purchase_timestamp"].min() + pd.DateOffset(days=start)
    stop_date = orders["order_purchase_timestamp"].min() + pd.DateOffset(days=stop)

    mask1 = orders["order_purchase_timestamp"] >= start_date
    mask2 = orders["order_purchase_timestamp"] <= stop_date
    mask = (mask1) & (mask2)
    customers_id = orders.loc[mask, "customer_unique_id"].tolist()

    mask = customers.index.isin(customers_id)
    customers = customers.loc[mask, :].sort_index()
    
    end_time = time() - start_time
    print("Fin du traitement")
    print(f"Temps écoule : {end_time :.1f}s")
    
    return customers