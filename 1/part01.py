#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Karolína Virágová, xvirag00

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""
from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any

def distance(a: np.array, b: np.array) -> np.array:
    """
    Výpočet Euklidovské vzdálenosti mezi dvěma poli souřadnic `a` a `b`.

    Parametry:
    a (np.array): První pole souřadnic.
    b (np.array): Druhé pole souřadnic, se stejným rozměrem jako `a`.

    Návratová hodnota:
    Výsledek výpočtu
    """

    # Vypočítáme rozdíl mezi souřadnicemi a a b
    differences = np.subtract(a, b)

    # Poté umocníme na druhou
    squared_diff = np.power(differences, 2)

    # Sečteme 
    sum_squared_diff = np.sum(squared_diff, axis=1)
    
    # Nakonec odmocníme, výsledek je euklidovská vzdálenost
    dist = np.sqrt(sum_squared_diff)

    return dist

def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    """
    Generování grafu funkcí f_a(x) = a^2 * sin(x) pro různé hodnoty parametru `a`.

    Tato funkce vytváří grafy funkcí f_a(x) = a^2 * sin(x) pro každý prvek v seznamu `a`.
    Grafy jsou vykresleny v rozsahu x od 0 do 6π. Výsledek lze uložit do souboru nebo
    zobrazit na obrazovce.

    Parametry:
    a (List[float]): List hodnot parametru `a`.
    show_figure (bool): Pokud je `True`, graf se zobrazí na obrazovce, pokud je zadaáno.
    save_path (Optional[str]): graf se uloží, pokud je zadáno.

    Návratová hodnota:
    Žádná. Funkce vykreslí graf a volitelně jej zobrazí nebo uloží.
    """
    # Definování rozsahu x od 0 do 6π
    x = np.linspace(0, 6 * np.pi, 200)
    sin_x = np.sin(x)
    a_array = np.array(a)[:, np.newaxis]

    # Generování matice výsledků f_a(x) = a^2 * sin(x)
    results = (a_array ** 2) * sin_x

    # Nastavení grafu
    plt.figure(figsize=(10, 6))

    # Vizualizace jednotlivých funkcí
    for i, f in enumerate(results):
        plt.plot(x, f, label=f'$y_{{{a[i]}}}(x)$')
        plt.fill_between(x, f, alpha=0.1) 

    # Nastavení popisků osy x a y
    plt.xlabel('$x$')
    plt.ylabel('$f_a(x)$')

    # Určení pozic na ose x pro tick marky (v násobcích π/2)
    tick_positions = np.arange(0, 6.5 * np.pi, np.pi / 2)

    # Vytvoření seznamu popisků pro tick marky v LaTeXu
    tick_labels = [
        r'$0$', r'$\frac{1}{2}\pi$', r'$\pi$', r'$\frac{3}{2}\pi$', r'$2\pi$',
        r'$\frac{5}{2}\pi$', r'$3\pi$', r'$\frac{7}{2}\pi$', r'$4\pi$',
        r'$\frac{9}{2}\pi$', r'$5\pi$', r'$\frac{11}{2}\pi$', r'$6\pi$',
        r'$6\frac{1}{2}\pi$'
    ]

    # Nastavení osy x: pozice ticků a odpovídající popisky
    plt.xticks(ticks=tick_positions, labels=tick_labels)

    # Nastavení osy y a x
    plt.ylim(-52, 52)
    plt.xlim(0, 6 * np.pi)

    # Přidání legendy nad graf
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(a))  # Legend nad grafem, uprostřed

    # Uložení grafu, pokud je specifikováno save_path
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    # Zobrazení grafu, pokud je show_figure True
    if show_figure:
        plt.show()

def plot_single_graph(ax, t: np.array, f: np.array, label: str):
    """
    Vykreslí jeden podgraf pro zadanou funkci.

    Parametry:
    ax : Objekt osy Matplotlib, na který se má vykreslit graf.
    t (np.array): Časový rozsah.
    f (np.array): Hodnoty funkce, které se mají vykreslit.
    label (str): Popisek y-osy.
    """
    ax.plot(t, f, color='blue')
    ax.set_ylabel(label, fontsize=10)
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks(np.arange(-0.8, 0.9, 0.4))
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 25))
    ax.set_xticklabels([])  # Skryje číslování na ose x


def plot_colored_sum(ax, t: np.array, f_sum: np.array, f1: np.array):
    """
    Vykreslí třetí podgraf pro součet funkcí s podmíněným barevným kódováním.

    Parametry:
    ax : Objekt osy Matplotlib, na který se má vykreslit graf.
    t (np.array): Časový rozsah.
    f_sum (np.array): Hodnoty součtu funkcí f1 a f2.
    f1 (np.array): Hodnoty funkce f1 pro porovnání s f_sum.
    """
    for i in range(1, len(t)):
        color = 'green' if f_sum[i] > f1[i] else ('red' if t[i] < 50 else 'orange')
        ax.plot(t[i-1:i+1], f_sum[i-1:i+1], color=color, linewidth=2)

    ax.set_ylabel(r'$f_1(t) + f_2(t)$', fontsize=10)
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks(np.arange(-0.8, 0.9, 0.4))
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 25))

def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    Generuje pokročilou vizualizaci sinusového signálu s třemi podgrafy:
    f1(t), f2(t) a jejich součtem s barevným kódováním.

    Parametry:
    show_figure (bool): graf se zobrazí na obrazovce, pokud je specifikováno.
    save_path (str): graf se uloží, pokud je specifikováno.
    
    Návratová hodnota:
    Žádná. Funkce vykreslí grafy a volitelně je zobrazí nebo uloží.
    """
    # Generování časového rozsahu a funkcí
    t = np.linspace(0, 100, 1000)
    f1 = 0.5 * np.cos(0.02 * np.pi * t)
    f2 = 0.25 * (np.sin(np.pi * t) + np.sin((2 / 3) * np.pi * t))
    f_sum = f1 + f2

    # Vytvoření figure a podgrafů
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharey=False)

    # Vykreslení prvních dvou podgrafů
    plot_single_graph(axs[0], t, f1, r'$f_1(t)$')
    plot_single_graph(axs[1], t, f2, r'$f_2(t)$')

    # Vykreslení třetího podgrafu s barevným kódováním
    plot_colored_sum(axs[2], t, f_sum, f1)

    # Uložení grafu, pokud je specifikováno `save_path`
    if save_path:
        fig.savefig(save_path)

    # Zobrazení grafu, pokud je `show_figure` True
    if show_figure:
        plt.show()

def parse_row(columns: List[str]) -> Dict[str, Any]:
    """
    Extrahuje a převádí hodnoty z řádku tabulky do slovníku.

    Args:
        columns (List[str]): Seznam textových hodnot z každého sloupce v řádku.

    Returns:
        Dict[str, Any]: Slovník s převáděnými hodnotami.
        Vyvolává ValueError, pokud převod selže.
    """

    if len(columns) < 7:
        raise ValueError("Nedostatečný počet sloupců v řádku.")

    # Získání názvu stanice
    position = columns[0].strip()
    try:
        # Převedení zeměpisné šířky/zeměpisné délky/nadmořské výšky  na float.
        lat_value = float(columns[2].split('°')[0].replace(',', '.'))
        long_value = float(columns[4].split('°')[0].replace(',', '.'))
        height_value = float(columns[6].replace(',', '.'))
    except ValueError as e:
        raise ValueError(f"Chyba při převodu hodnot: {e}")

    return {
        'position': position,
        'lat': lat_value,
        'long': long_value,
        'height': height_value
    }


def download_data() -> Dict[str, List[Any]]:
    """
    Stáhne a zpracuje data geografických stanic z webové stránky.

    Returns:
        Dict[str, List[Any]]: Slovník obsahující čtyři položky: 'positions', 'lats', 'longs', 'heights'.
    """
    # URL stránka obsahující data stanic
    url = "https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html"

    # Stažení obsahu, nastaveni kódování
    response = requests.get(url)
    response.encoding = 'utf-8'

    if response.status_code != 200:
        raise Exception(f"Selhalo načítání dat. Kód stavu: {response.status_code}")
    
    # Parsování obsahu, najití tabulek
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table')
    
    if len(tables) < 2:
        raise Exception("Druhá tabulka nebyla nalezena na stránce.")
    
    # Výběr druhé tabulky
    table = tables[1]

    # Inicializace slovníku s prázdnými seznamy
    data = {
        'positions': [],
        'lats': [],
        'longs': [],
        'heights': []
    }
    
    # Iterace přes řádky tabulky, přeskočení hlavičky
    rows = table.find_all('tr')
    for row in rows[1:]:
        columns = [col.text for col in row.find_all('td')]
        parsed_data = parse_row(columns)
        
        if parsed_data is None:
            continue  # Přeskočíme řádky, kde se převod nezdařil
        
        # Uložení hodnot do slovníku
        data['positions'].append(parsed_data['position'])
        data['lats'].append(parsed_data['lat'])
        data['longs'].append(parsed_data['long'])
        data['heights'].append(parsed_data['height'])

    return data

if __name__ == "main":
    generate_graph([7, 4, 3])
    generate_sinus()
