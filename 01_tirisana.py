import pandas as pd

# Ielādējam datu kopu no CSV
file = pd.read_csv("dati/penguins.csv")
initial_row_count = len(file)

# Noņemam dublējošās rindas un aprēķinām noņemto rindu skaitu
before = len(file)
file = file.drop_duplicates()
duplicates_removed = before - len(file)

# Norādām kolonnas, kur ir skaitliskā vērtībā
numeric_columns = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

# Pārveidojam kolonnu vērtības skaitliskā formātā.Nederīgās vērtības tiek aizstātas ar trūkstošām vērtībām
for i in numeric_columns:
    file[i] = pd.to_numeric(
        file[i], errors="coerce"
    )
    
# Izņemam rindas, kurās trūkst kāds no nepieciešamajiem mērījumiem
before = len(file)
file = file.dropna(subset=numeric_columns)
rows_without_measurements = before - len(file)

# Saskaitām nezināmās dzimuma vērtības un aizstājam tās ar "unknown"
unknown_sex_count = file["sex"].isna().sum()
file["sex"] = file["sex"].fillna("unknown")

# Saglabājam iztīrītos datus jaunā CSV
file.to_csv("tirs.csv", index=False)

#Kopsavilkums
print("Initial rows:", initial_row_count)
print("Duplicates removed:", duplicates_removed)
print("Rows without measurements removed:", rows_without_measurements)
print("Unknown sex values filled:", unknown_sex_count)
print("Final rows:", len(file))