#!/usr/bin/python3.10
# coding=utf-8

"""
IZV část 3 projektu- Vizualizace geografických dat.

Autor: Karolína Virágová, xvirag00.
"""

import seaborn as sns
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import MultiPoint
from sklearn.cluster import KMeans
import numpy as np
import geopandas as gpd


def make_geo(
    df_accidents: pd.DataFrame, df_locations: pd.DataFrame
) -> geopandas.GeoDataFrame:
    """
    Vytváří GeoDataFrame z dvou DataFramů obsahující
    informace o nehodách a jejich lokalitách.

    Parametry:
    - df_accidents: DataFrame obsahující informace o nehodách
    - df_locations: DataFrame obsahující geografické souřadnice

    Návratová hodnota:
    - GeoDataFrame obsahující informace o nehodách se souřadnicemi.
    """
    # Spojení dataframu
    df = pd.merge(df_accidents, df_locations, on="p1", how="inner")

    # Odstranění řádků, kde chybí souřadnice
    df = df.dropna(subset=["d", "e"])

    # Kontrola, zda je potřeba prohodit hodnoty mezi souřadnicemi (d a e),
    swapped_condition = df["d"] < df["e"]
    if swapped_condition.any():
        # Prohození hodnot v těchto sloupcích
        df.loc[swapped_condition, ["d", "e"]] = df.loc[
            swapped_condition, ["e", "d"]
        ].values

    # Filtrace podle regionu
    df = df[df["region"] == "JHM"]

    # Vytvoření GeoDataFrame
    gdf = geopandas.GeoDataFrame(
        df,
        geometry=geopandas.points_from_xy(df["d"], df["e"]),
        crs="EPSG:5514"
    )

    gdf = gdf.to_crs(epsg=4326)
    return gdf


def plot_geo(
    gdf: geopandas.GeoDataFrame, fig_location: str = None,
    show_figure: bool = False
):
    """
    Vykresluje grafy nehod, které byly zaviněné alkoholem, ve vybraném kraji
    pro měsíce leden a červenec.

    Parametry:
    - gdf: GeoDataFrame obsahující data o nehodách
    - fig_location: Cesta k souboru, kam se má uložit obrázek
    - show_figure: Pokud je True, zobrazí graf v okně
    """
    # Ověření, zda DataFrame obsahuje sloupec 'date'
    if "date" in gdf.columns:
        gdf["month"] = gdf["date"].dt.month
    else:
        raise ValueError(
            "GeoDataFrame musí obsahovat sloupec 'date' ve formátu YYYY-MM-DD."
        )

    # Filtrování nehod, kde byl řidič pod vlivem alkoholu
    alcohol_accidents = gdf[gdf["p11"] >= 4]

    # Filtrování nehod, které se staly v lednu
    accidents_january = alcohol_accidents[alcohol_accidents["month"] == 1]

    # Filtrování nehod, které se staly v červenci
    accidents_july = alcohol_accidents[alcohol_accidents["month"] == 7]

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Vykreslení lednových nehod
    sns.scatterplot(
        x=accidents_january.geometry.x,
        y=accidents_january.geometry.y,
        ax=axes[0],
        color="blue",
        s=30,
        label="Nehody",
        marker="x",
        linewidths=2,
    )

    # Přidání podkladové mapy (OpenStreetMap)
    ctx.add_basemap(
        axes[0],
        crs=gdf.crs.to_string(),
        source=ctx.providers.OpenStreetMap.Mapnik,
        alpha=0.9,
    )
    axes[0].set_title("JHM kraj- Nehody pod vlivem alkoholu (Leden)")

    # Vykreslení červencových nehod
    sns.scatterplot(
        x=accidents_july.geometry.x,
        y=accidents_july.geometry.y,
        ax=axes[1],
        color="red",
        s=20,
        label="Nehody",
        marker="x",
        linewidths=2,
    )
    # Přidání podkladové mapy (OpenStreetMap)
    ctx.add_basemap(
        axes[1],
        crs=gdf.crs.to_string(),
        source=ctx.providers.OpenStreetMap.Mapnik,
        alpha=0.9,
    )
    axes[1].set_title("JHM kraj- Nehody pod vlivem alkoholu (Červenec)")

    if fig_location:
        plt.savefig(fig_location, dpi=300, bbox_inches="tight")
    if show_figure:
        plt.show()
    plt.close()


def plot_cluster(
    gdf: gpd.GeoDataFrame, fig_location: str = None,
    show_figure: bool = False
):
    """
    Vykresluje nehody v kraji a znázorňuje oblasti nehod zaviněných zvěří.

    Parametry:
    - gdf: geopandas.GeoDataFrame obsahující data o nehodách
    - fig_location: Cesta k souboru, kam se má uložit výstup
    - show_figure: Pokud je True, zobrazí graf v okně
    """
    if gdf.empty:
        raise ValueError("GeoDataFrame is empty")

    if gdf.crs is None or gdf.crs.to_string() != "EPSG:3857":
        gdf = gdf.to_crs("EPSG:3857")

    # Filtr pro nehody s lesní zvěří
    gdf = gdf[gdf["p10"] == 4]

    coords = np.array([(point.x, point.y) for point in gdf.geometry])

    # Použijeme KMeans pro shlukování
    kmeans = KMeans(n_clusters=7, random_state=42).fit(coords)
    gdf["cluster"] = kmeans.labels_

    # Vypočítáme velikost každého shluku
    cluster_sizes = gdf["cluster"].value_counts().sort_index()

    norm = plt.Normalize(vmin=cluster_sizes.min(), vmax=cluster_sizes.max())
    cmap = plt.cm.viridis

    cluster_polygons = []
    valid_clusters = []

    for cluster_id in cluster_sizes.index:
        cluster_points = gdf[gdf["cluster"] == cluster_id].geometry
        if len(cluster_points) > 2:
            cluster_polygon = MultiPoint(
                [p for p in cluster_points]
                ).convex_hull
            cluster_polygons.append((cluster_id, cluster_polygon))
            valid_clusters.append(cluster_id)

    gdf = gdf[gdf["cluster"].isin(valid_clusters)]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    gdf.plot(ax=ax, color="red", markersize=10, label="Nehody")

    # Podbarvení shluků s barvou odpovídající velikosti shluku
    for cluster_id, polygon in cluster_polygons:
        cluster_size = cluster_sizes[cluster_id]
        color = cmap(norm(cluster_size))
        patch = gpd.GeoSeries([polygon], crs=gdf.crs)
        patch.plot(
            ax=ax,
            color=color,
            alpha=0.5,
            edgecolor="black",
            linewidth=0.7,
            label=f"Cluster {cluster_id}",
        )

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label="Počet nehod v oblasti")
    ax.set_title("Nehody v JHM kraji zaviněné lesní zvěří")

    if fig_location:
        plt.savefig(fig_location, dpi=300, bbox_inches="tight")
    if show_figure:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    df_accidents = pd.read_pickle("accidents.pkl.gz")
    df_locations = pd.read_pickle("locations.pkl.gz")
    gdf = make_geo(df_accidents, df_locations)

    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)

    # testovani splneni zadani
    import os

    assert os.path.exists("geo1.png")
    assert os.path.exists("geo2.png")
