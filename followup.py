import pandas as pd
from cleverminer import cleverminer
import io
import sys

#Jsou vážné dopravní nehody o víkendu častější než ve všední dny v závislosti 
# na počasí a přítomnosti jakéhokoliv typu dopravní infrastruktury?“
# 1. Načtení dat
df = pd.read_csv('newdataset_clean.csv', delimiter=';', engine='python', on_bad_lines='skip')
df.columns = df.columns.str.strip()

# 2. Zpracování data a vytvoření atributu "day_type"
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df = df.dropna(subset=['Start_Time'])

df['day_type'] = df['Start_Time'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

def categorize_weather_numeric(condition):
    # 1) nejprve očistíme od whitespace
    cleaned = str(condition).strip()
    
    # 2) namapujeme na hlavní kategorie
    category_mapping = {
        # CLEAR
        'Fair': 'Clear',
        'Fair / Windy': 'Clear',

        # CLOUDY
        'Partly Cloudy': 'Cloudy',
        'Mostly Cloudy': 'Cloudy',
        'Cloudy': 'Cloudy',
        'Cloudy / Windy': 'Cloudy',
        'Mostly Cloudy / Windy': 'Cloudy',
        'Partly Cloudy / Windy': 'Cloudy',
            

        # RAIN
        'Rain': 'Rain',
        'Heavy Rain': 'Rain',
        'Light Rain': 'Rain',
        'Light Rain / Windy': 'Rain',
        'Light Drizzle': 'Rain',
        'Drizzle': 'Rain',
        'Drizzle / Windy': 'Rain',
        'Light Drizzle / Windy': 'Rain',
        'Heavy Rain / Windy': 'Rain',
        'Rain / Windy': 'Rain',
        'Light Rain Shower': 'Rain',
        'Heavy Drizzle': 'Rain',
        'Showers in the Vicinity': 'Rain',
        'Freezing Rain': 'Rain',
        'Freezing Rain / Windy': 'Rain',
        'Light Freezing Rain': 'Rain',
        'Freezing Drizzle': 'Rain',
        'Light Freezing Drizzle': 'Rain',

        # SNOW
        'Snow': 'Snow',
        'Light Snow': 'Snow',
        'Heavy Snow': 'Snow',
        'Wintry Mix': 'Snow',
        'Snow and Sleet': 'Snow',
        'Blowing Snow': 'Snow',
        'Light Snow / Windy': 'Snow',
        'Snow / Windy': 'Snow',
        'Heavy Snow / Windy': 'Snow',
        'Wintry Mix / Windy': 'Snow',
        'Light Snow and Sleet': 'Snow',
        'Snow and Sleet / Windy': 'Snow',
        'Blowing Snow / Windy': 'Snow',
        'Sleet': 'Snow',
        'Sleet / Windy': 'Snow',
        'Light Snow Shower': 'Snow',

        # STORM
        'T-Storm': 'Storm',
        'Thunder': 'Storm',
        'Heavy T-Storm': 'Storm',
        'Thunder in the Vicinity': 'Storm',
        'Thunder / Wintry Mix': 'Storm',
        'Thunder / Wintry Mix / Windy': 'Storm',
        'T-Storm / Windy': 'Storm',
        'Thunder / Windy': 'Storm',
        'Hail': 'Storm',
        'Light Rain with Thunder': 'Storm',
        'Light Snow with Thunder': 'Storm',

        # FOG
        'Fog': 'Fog',
        'Haze': 'Fog',
        'Mist': 'Fog',
        'Shallow Fog': 'Fog',
        'Patches of Fog': 'Fog',
        'Haze / Windy': 'Fog',
        'Fog / Windy': 'Fog',
        'Drizzle and Fog': 'Fog',
        'Widespread Dust': 'Fog',
        'Blowing Dust': 'Fog',
        'Blowing Dust / Windy': 'Fog',
        'N/A Precipitation': 'Fog'
    }

    category = category_mapping.get(cleaned, 'Unknown')

    # 3) čísla pro ordinalitu
    label_to_number = {
        'Clear': 1,
        'Cloudy': 2,
        'Rain': 3,
        'Snow': 4,
        'Fog': 5,
        'Storm': 6,
        'Unknown': None
    }

    return label_to_number[category]

 
df['Weather_numeric'] = df['Weather_Condition'].apply(categorize_weather_numeric)
# případně odfiltrovat Unknown:
df = df[df['Weather_numeric'].notna()]

# 3. Výběr relevantních sloupců
infra_cols = [
    'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
    'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming',
    'Traffic_Signal', 'Turning_Loop'
]


relevant_cols = infra_cols + ['Weather_numeric', 'Severity', 'day_type']
df = df[relevant_cols]

# 4. Čištění hodnot
for col in infra_cols:
    df[col] = df[col].astype(str).str.upper().str.strip().map(lambda x: 'Yes' if x == 'PRAVDA' else 'No')

df = df.dropna(subset=['Severity', 'day_type'])


# 5. Cílová proměnná
df['severity_level'] = df['Severity'].apply(lambda x: 'High' if int(x) >= 3 else 'Low')

# 6. Spuštění SD4ftMiner
clm = cleverminer(
    df=df,
    proc='SD4ftMiner',
    quantifiers={
        'Base1': 300,
        'Base2': 300,
        'Ratiopim': 1.1
    },
    ante={
        'attributes': [
        {'name': 'Weather_numeric', 'type': 'seq', 'minlen': 1, 'maxlen': 2},
    ] + [
        {'name': col, 'type': 'subset', 'minlen': 1, 'maxlen': 1} for col in infra_cols
    ],
            
        'minlen': 1,
        'maxlen': 2,
        'type': 'con'
    },
    succ={
        'attributes': [
            {'name': 'severity_level', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    },
    frst={
        'attributes': [
            {'name': 'day_type', 'type': 'one', 'value': 'Weekday'}
        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    },
    scnd={
        'attributes': [
            {'name': 'day_type', 'type': 'one', 'value': 'Weekend'}
        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    }
)

# 7. Výstup
output = io.StringIO()
sys.stdout = output

print("=== SD4ft-Miner – Víkend vs. Pracovní dny a závažnost nehody ===\n")
clm.print_summary()
print("\n=== Pravidla ===\n")
clm.print_rulelist()
if clm.rulelist:
    print("\n=== První pravidlo ===\n")
    clm.print_rule(1)

sys.stdout = sys.__stdout__

# 8. Uložení
with open("sd4ftminer_question_followup_weekday_weekend.txt", "w", encoding="utf-8") as f:
    f.write(output.getvalue())

print("✅ Výstup uložen jako sd4ftminer_question_followup_weekday_weekend.txt")
