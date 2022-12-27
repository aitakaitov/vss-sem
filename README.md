# Prerekvizity

Stačí spustit příkaz

<code>pip install -r requirements.txt</code>

# Použití

Program lze spustit příkazem

<code>python main.py --dataset_dir data/MDP</code>

Pro specifikaci výstupní složky navíc s parametrem <code>--results_dir ...</code>. Bez tohoto parametru jsou výsledky
uloženy do složky <code>results</code>.

Složka s výsledky obsahuje <code>csv</code> soubor pro každý dataset, ve kterém jsou pro všechny 
tři modely jejich metriky.

# Struktura
* <code>csv_utils.py</code> - obsahuje funkce pro výpis výsledků do CSV souborů
* <code>dataset_utils.py</code> - obsahuje funkce a třídy pro práci se soubory datasetu
* <code>fit.py</code> - obsahuje trénovací a validační logiku
* <code>flexible_bayes.py</code> - obsahuje implementaci Flexibilního bayese
* <code>metrics.py</code> - obsahuje výpočet metrik
* <code>main.py</code> - obsahuje parsování agrumentů, načítání datasetů a spouštění testů
