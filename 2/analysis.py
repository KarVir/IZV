#!/usr/bin/env python3.12
# coding=utf-8

"""
IZV část 2 projektu.

Autor: Karolína Virágová, xvirag00.

Tento modul obsahuje funkce pro analýzu dat o dopravních nehodách.
Používá knihovny matplotlib, pandas a seaborn pro vizualizaci
a zpracování dat.
"""

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import zipfile

# muzete pridat libovolnou zakladni knihovnu ci
# knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz


# Ukol 1: nacteni dat ze ZIP souboru
def load_data(filename: str, ds: str) -> pd.DataFrame:
    """
    Načte data ze zadaného ZIP souboru a spojí je do jednoho DataFrame.

    Funkce zpracuje data z adresářů 2023 a 2024 uvnitř ZIP souboru
    a spojí je do jednoho DataFrame.

    Args:
        filename (str): Cesta k ZIP souboru.
        ds (str): Název souboru (bez prefixu "I").

    Returns:
        pd.DataFrame: DataFrame obsahující data z obou let.
    """
    # Otevření zip souboru
    with zipfile.ZipFile(filename, 'r') as zip_file:
        # Definování cest k obou souborům
        paths_to_check = [f"2023/I{ds}.xls", f"2024/I{ds}.xls"]

        # Inicializace prázdného dataframu
        dataframes = []

        # Procházení cesty souborům  a načtení existujícího souboru
        for path in paths_to_check:
            if path in zip_file.namelist():
                with zip_file.open(path) as file:
                    # Načtení souboru do dataframu
                    df = pd.read_html(file, encoding="cp1250")[0]
                    # Odebere nepojmenované sloupce
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                    # Přidání obsahu souboru do dataframu
                    dataframes.append(df)

        # Jestli nejsou nalezeny žádné souboru
        if not dataframes:
            raise FileNotFoundError(f"No files {df} found in the ZIP archive.")

        # Zřetězí všechny načtené soubory
        combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


# Ukol 2: zpracovani dat
def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Zpracuje DataFrame a připraví jej pro další analýzu.

    Vytvoří sloupec 'date' na základě hodnot ze sloupce 'p2a',
    které jsou převedeny na datetime. Přidá také sloupec 'region'
    na základě hodnot ve sloupci 'p4a'. Funkce dále odstraní
    duplicitní řádky na základě sloupce 'p1'.

    Args:
        df (pd.DataFrame): obsahuje data o nehodách.
        verbose (bool): Pokud je True, vypíše velikost výsledného DataFrame.

    Returns:
        pd.DataFrame: Upravený DataFrame se sloupci 'date' a 'region'.
    """
    # Vytvoření kopie dataframu
    df2 = df.copy()

    # Vytvoření sloupce date z původního sloupce p2a a konverze na formát data
    df2['date'] = pd.to_datetime(df['p2a'], dayfirst=True, errors="coerce")

    # Vytvoření sloupce region
    region_map = {
        0: "PHA", 1: "STC", 2: "JHC", 3: "PLK", 4: "ULK", 5: "HKK",
        6: "JHM", 7: "MSK", 14: "OLK", 15: "ZLK", 16: "VYS",
        17: "PAK", 18: "LBK", 19: "KVK"
    }
    df2['region'] = df2['p4a'].map(region_map)

    # Odstranění duplikátů na základě sloupce p1
    df2 = df2.drop_duplicates(subset=['p1'])

    # Je-li verbose pravdivé, vypočítá se a vypíše se velikost dataframe
    if verbose:
        size_bytes = df2.memory_usage(deep=True).sum()
        size_mb = size_bytes / 1_000_000
        print(f"new_size={size_mb:.1f} MB")

    return df2


# Ukol 3: počty nehod v jednotlivých regionech podle stavu vozovky
def plot_state(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
    """
    Zpracuje DataFrame obsahující data o nehodách a jejich následcích.

    Vytvoři grafy podle stavu vozovky (např. suchý, mokrý), který zobrazuje:
        - počet nehod podle regionu,

    Args:
        df (pd.DataFrame): obsahující informace o nehodách.
        df_consequences (pd.DataFrame): obsahuje informace o následcích nehod.
        fig_location (str, optional): Cesta k souboru, kam bude graf uložen.
        show_figure (bool, optional): Pokud je True, graf bude zobrazen.

    Returns:
        None: Funkce nezpůsobuje vrácení hodnoty, pouze vykresluje grafy.
    """
    # Vytvoření sloupce state na základě sloupce p16
    p16_map = {
        1: "povrch suchý",
        2: "povrch suchý",
        3: "povrch mokrý",
        4: "na vozovce je bláto",
        5: "na vozovce je náledí, ujetý sníh ",
        6: "na vozovce je náledí, ujetý sníh "
    }
    df['state'] = df['p16'].map(p16_map)

    # Agregace počtu nehod podle regionu a stavu vozovky
    region_state_counts = df.groupby(['region', 'state']).size()
    region_state_counts = region_state_counts.unstack(fill_value=0)
    # Nastavení velikosti obrázku a uspořádání 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Vytvoření jednotlivých grafů pro každý stav vozovky
    states = region_state_counts.columns

    for i, state in enumerate(states):
        ax = axes[i // 2, i % 2]
        palette = sns.color_palette("Set2", len(region_state_counts))
        region_state_counts[state].plot(kind='bar', ax=ax, color=palette)

        ax.set_title(f"Stav povrchu vozovky: {state}")

        # Nastavení os pouze u vybraných subplotů
        if i < 2:  # První řádek
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Kraj")

        if i % 2 == 0:  # První sloupec
            ax.set_ylabel("Počet nehod")
        else:
            ax.set_ylabel("")

        fig.suptitle("Počty nehod podle stavu vozovky", fontweight='bold')
        ax.set_xticklabels(region_state_counts.index, rotation=45)
        ax.grid(True)

    #  Uložení grafu do souboru
    if fig_location:
        plt.savefig(fig_location, bbox_inches='tight')
    # Zobrazení grafu
    if show_figure:
        plt.show()
    plt.close()


# Ukol4: alkohol a následky v krajích
def plot_alcohol(df: pd.DataFrame, df_consequences: pd.DataFrame,
                 fig_location: str = None, show_figure: bool = False):
    """
    Zpracuje DataFrame obsahující data o nehodách.

    Vytvoří grafy, které zobrazují počet nehod podle regionu,
    typu osoby (řidič/spolujezdec) a úrovně zranění
    (úmrtí, těžké zranění, lehké zranění, bez zranění).

    Args:
        df (pd.DataFrame): obsahuje informace o nehodách.
        df_consequences (pd.DataFrame): obsahuje informace o následcích nehod.
        fig_location (str): Cesta k souboru, kam bude graf uložen.
        show_figure (bool): Pokud je True, graf bude zobrazen.

    Returns:
        None: Funkce nezpůsobuje vrácení hodnoty, pouze vykresluje grafy.
    """
    # Step 1: Join dataframes on the 'p1' column (1:n relationship)
    merged_df = df.merge(df_consequences, on='p1', how='left')

    # Kontrola zda existují požadované sloupce
    required_columns = ['p11', 'region', 'p59a', 'p59g']
    for col in required_columns:
        if col not in merged_df.columns:
            raise ValueError(f"Column '{col}' is missing from the data.")

    # Filtrování nehod jestli byla způsobena pod vlivem alkoholu (p11 >= 3)
    merged_df = merged_df[merged_df['p11'] >= 3]

    # Určení zda byl zraněn řidič (p59a == 1) nebo spolujezdec (p59a != 1)
    merged_df['person_type'] = merged_df['p59a'].apply(
        lambda x: 'Řidič' if x == 1 else 'Spolujezdec'
        )

    # Filtrování nehod, které mají definované zranění (p59g)
    merged_df = merged_df[merged_df['p59g'].notna()]

    # Mapa úrovní zranění
    injury_level_map = {
        1: "Úmrtí",
        2: "Těžké zranění",
        3: "Lehké zranění",
        4: "Bez zranění"
    }

    # Seskupení dat podle regionu, typu osoby a úrovně zranění
    injury_counts = merged_df.groupby(
        ['region', 'person_type', 'p59g']
        ).size().reset_index(name='count')
    injury_counts['injury_level'] = injury_counts['p59g'].map(injury_level_map)

    # Určení pořadí úrovní zranění
    injury_levels = ["Bez zranění", "Lehké zranění", "Těžké zranění", "Úmrtí"]

    # Vytvoření subgrafů (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Menší velikost obrázku
    axs = axs.flatten()

    # Inicializace handles a labels pro legendu
    handles, labels = None, None

    for i, level in enumerate(injury_levels):
        ax = axs[i]

        # Výběr dat pro konkrétní úroveň zranění
        level_data = injury_counts[injury_counts['injury_level'] == level]

        # Nastavení názvu grafu
        ax.set_title(f"Následky nehody: {level}")

        # Vykreslení grafu
        sns.barplot(
            data=level_data,
            x="region",
            y="count",
            hue="person_type",
            ax=ax, palette="muted",
            dodge=True
        )

        # Odebrání osových popisků (x, y) explicitně
        if i < 2:  # První řada grafů (odebrat "Region")
            ax.set_xlabel(None)
        else:
            ax.set_xlabel("Kraj")

        if i % 2 == 1:
            ax.set_ylabel(None)
        else:
            ax.set_ylabel("Počet nehod pod vlivem")

        # Uložení handles a labels pro legendu (pouze z prvního subplotu)
        if handles is None and labels is None:
            handles, labels = ax.get_legend_handles_labels()

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Odebrání duplicitních legend ze subplotů
    for ax in axs:
        ax.get_legend().remove()

    # Uspořádání panelů - grafy blíže u sebe
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    fig.suptitle(
        "Alkohol a následky v jednotlivých krajích",
        fontweight='bold'
    )

    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc='center left',
            bbox_to_anchor=(0.95, 0.5),
            title="Typ účastníka")

    # Uložení nebo zobrazení grafu
    if fig_location:
        plt.savefig(fig_location, bbox_inches='tight')
    if show_figure:
        plt.show()

    plt.close()


# Ukol 5: Druh nehody (srážky) v čase
def plot_type(df: pd.DataFrame, fig_location: str = None,
              show_figure: bool = False):
    """
    Zpracuje DataFrame obsahující data o nehodách.

    Vytvoří grafy, které zobrazují počet nehod podle regionu
    a typu kolize (např. s vozidlem, s chodcem) v čase (měsíčně).

    Args:
        df (pd.DataFrame): DataFrame obsahující informace o nehodách.
        fig_location: Cesta k souboru, kam bude graf uložen.
        show_figure: Pokud je True, graf bude zobrazen na obrazovce.

    Returns:
        None: Funkce nezpůsobuje vrácení hodnoty, pouze vykresluje grafy.
    """
    # Vybrané kraje
    selected_regions = ['JHM', 'PHA', 'STC', 'PLK']
    df = df[df['region'].isin(selected_regions)]

    # Mapování hodnot 'p6' na textové popisy
    collision_types = {
        1: "s jedoucím nekolejovým vozidlem",
        2: "s vozidlem zaparkovaným, odstaveným",
        3: "s pevnou překážkou",
        4: "s chodcem",
        5: "s lesní zvěří",
        6: "s domácím zvířetem",
        7: "s vlakem",
        8: "s tramvají"
    }
    df['collision_type'] = df['p6'].map(collision_types)

    # Konverze datumu na datetime
    df['date'] = pd.to_datetime(df['date'])

    # Pivotování dat pro denní počty nehod podle typu srážky
    daily_counts = df.pivot_table(
        index=['date', 'region'],
        columns='collision_type',
        values='p6',
        aggfunc='count',
        fill_value=0
    ).reset_index()

    # Resampling na měsíční počty
    daily_counts.set_index('date', inplace=True)
    monthly_counts = daily_counts.groupby('region').resample('M').sum()
    monthly_counts.reset_index(inplace=True)

    # Přeformátování pro stackovaný formát
    monthly_counts = monthly_counts.melt(
        id_vars=['date', 'region'],
        var_name='collision_type',
        value_name='count'
    )

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=False)
    axs = axs.flatten()

    handles, labels = [], []

    for i, region in enumerate(selected_regions):
        ax = axs[i]
        region_data = monthly_counts[monthly_counts['region'] == region]

        for collision_type in region_data['collision_type'].unique():
            subset = region_data[
                region_data['collision_type'] == collision_type
                ]
            line, = ax.plot(
                subset['date'],
                subset['count'],
                label=collision_type
            )

            if collision_type not in labels:
                handles.append(line)
                labels.append(collision_type)

        ax.set_xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2024-10-01'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))

        # Nastavení osy Y
        ax.set_ylabel("Počet nehod")
        ax.set_title(f"Kraj: {region}")

    fig.suptitle(
        "Počet nehod podle druhu srážky v čase",
        fontsize=16, fontweight='bold'
    )

    fig.legend(
        handles, labels,
        loc='center right',
        bbox_to_anchor=(1.25, 0.5),
        title="Druh nehody"
    )

    if fig_location:
        plt.savefig(fig_location, bbox_inches='tight')

    if show_figure:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni
    # funkce.

    df = load_data("data_23_24.zip", "nehody")
    df_consequences = load_data("data_23_24.zip", "nasledky")
    df2 = parse_data(df, True)

    plot_state(df2, "01_state.png")
    plot_alcohol(df2, df_consequences, "02_alcohol.png", True)
    plot_type(df2, "03_type.png")
# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku
