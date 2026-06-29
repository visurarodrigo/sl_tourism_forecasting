import pandas as pd
import os
import re

# Bulletproof path finding
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

input_path = os.path.join(project_root, 'Data', '01_raw', 'sl_tourism_country_wise.csv')
output_dir = os.path.join(project_root, 'Data', '02_processed')
output_path = os.path.join(output_dir, 'sl_tourism_country_wise_clean.csv')

os.makedirs(output_dir, exist_ok=True)

print("Loading country-wise data...")
df = pd.read_csv(input_path, parse_dates=['Date'])

print(f"Original unique countries: {df['Country'].nunique()}")

# ==========================================
# FLAGSHIP MOVE: COMPREHENSIVE ENTITY RESOLUTION
# ==========================================

# 1. Exact match dictionary for known variations
exact_corrections = {
    # A
    'Afghanistan, Islamic Republic Of': 'Afghanistan',
    'Albania, Republic Of': 'Albania',
    'Algeria, People\'S Democratic Republic Of': 'Algeria',
    'Andorra, Principality Of': 'Andorra',
    'Angola, Republic Of': 'Angola',
    'Antigua & Barbuda': 'Antigua And Barbuda',
    'Armenia, Republic Of': 'Armenia',
    'Azerbaijan, Republic Of': 'Azerbaijan',
    
    # B
    'Bahamas, Commonwealth Of The': 'Bahamas',
    'Bahrain, Kingdom Of': 'Bahrain',
    'Bangladesh, People\'S Republic Of': 'Bangladesh',
    'Belarus, Republic Of': 'Belarus',
    'Bhutan, Kingdom Of': 'Bhutan',
    'Bosnia': 'Bosnia And Herzegovina',
    'Bosnia & Herzegovina': 'Bosnia And Herzegovina',
    'Botswana, Republic Of': 'Botswana',
    'Brazil, Federative Republic Of': 'Brazil',
    'Brunei': 'Brunei Darussalam',
    'Brunei, Nation Of': 'Brunei Darussalam',
    'Burkina': 'Burkina Faso',
    
    # C
    'Cambodia, Kingdom Of': 'Cambodia',
    'Chile, Republic Of': 'Chile',
    'China, People\'S Republic Of': 'China',
    'Colombia, Republic Of': 'Colombia',
    'Congo': 'Congo, Republic Of The',
    'Congo The Dem Repof The': 'Democratic Republic Of Congo',
    'Congo, Democratic Republic Of The': 'Democratic Republic Of Congo',
    'Congo, Republic Of The': 'Congo, Republic Of The',
    'Congo, Republic Of.': 'Congo, Republic Of The',
    'Congo, The Democratic Republic': 'Democratic Republic Of Congo',
    'Costa Rica, Republic Of': 'Costa Rica',
    'Cote Divoire': 'Cote D\'Ivoire',
    'Cote D\'Ivoire, Republic Of': 'Cote D\'Ivoire',
    'Cote D\'Lvoire': 'Cote D\'Ivoire',
    'Cuba, Republic Of': 'Cuba',
    'Cyprus, Republic Of': 'Cyprus',
    
    # D
    'Democratic Republic Of  Congo': 'Democratic Republic Of Congo',
    'Denmark, Kingdom Of': 'Denmark',
    'Djibouti, Republic Of': 'Djibouti',
    'Dominica, Commonwealth Of': 'Dominica',
    
    # E
    'Ecuador, Republic Of': 'Ecuador',
    'Egypt, Arab Republic Of': 'Egypt',
    'El Salvador, Republic Of': 'El Salvador',
    'Eritrea, State Of': 'Eritrea',
    'Ethiopia, Federal Democratic Republic Of': 'Ethiopia',
    
    # F
    'Fiji, Republic Of': 'Fiji',
    'France (French Republic)': 'France',
    
    # G
    'Gambia': 'The Gambia',
    'Gambia, The': 'The Gambia',
    'Ghana, Republic Of': 'Ghana',
    'Guatemala, Republic Of': 'Guatemala',
    'Guatemala Guinea': 'Guatemala',  # Corruption fix
    'Guinea- Bissau': 'Guinea-Bissau',
    'Guinea-Bissau Guinea': 'Guinea-Bissau',  # Corruption fix
    'Guyana, Cooperative Republic Of': 'Guyana',
    
    # H
    'Haiti, Republic Of': 'Haiti',
    'Honduras, Republic Of': 'Honduras',
    
    # I
    'Indonesia, Republic Of': 'Indonesia',
    'Iraq, Republic Of': 'Iraq',
    'Israel, State Of': 'Israel',
    
    # J
    'Jordan, The Hashemite Kingdom Of': 'Jordan',
    
    # K
    'Kazakhstan, Republic Of': 'Kazakhstan',
    'Kenya, Republic Of': 'Kenya',
    'Kosovar': 'Kosovo',
    'Kuwait, State Of': 'Kuwait',
    
    # L
    'Lao Peoples': 'Lao People\'S Democratic Republic',
    'Lao Peoples Dem. Rep.': 'Lao People\'S Democratic Republic',
    'Lao Peoples Dem.Rep.': 'Lao People\'S Democratic Republic',
    'Lao, People\'S Democratic': 'Lao People\'S Democratic Republic',
    'Lao, People\'S Democratic Republic': 'Lao People\'S Democratic Republic',
    'Libya(Libyan Arab Jamahir': 'Libya',
    'Libya(Libyan Arab Jamahir)': 'Libya',
    'Libya, State Of': 'Libya',
    'Liechtenstein, Principality Of': 'Liechtenstein',
    
    # M
    'Macedonia, Republic Of': 'Macedonia',
    'Maldives, Republic Of': 'Maldives',
    'Mali, Republic Of': 'Mali',
    'Marshall Islands, Republic Of The': 'Marshall Islands',
    'Mauritania, Islamic Republic Of': 'Mauritania',
    'Mauritius, Republic Of': 'Mauritius',
    'Micronesia, Federated States Of': 'Micronesia',
    'Moldova, Republic Of': 'Moldova',
    'Monaco, Principality Of': 'Monaco',
    'Morocco, Kingdom Of': 'Morocco',
    'Mozambique, Republic Of': 'Mozambique',
    'Myanmar, Republic Of The Union Of': 'Myanmar',
    
    # N
    'Nepal, Federal Democratic Republic Of': 'Nepal',
    'Netherland Antilles': 'Netherlands Antilles',
    'Niger,Republic Of': 'Niger',
    'Nigeria, Federal Republic Of': 'Nigeria',
    'North Korea': 'Korea, Democratic People\'S Republic Of',
    'Norway, Kingdom Of': 'Norway',
    
    # O
    'Oman, Sultanate Of': 'Oman',
    
    # P
    'Pakistan, Islamic Republic Of': 'Pakistan',
    'Panama, Republic Of': 'Panama',
    'Peru, Republic Of': 'Peru',
    'Philippines, Republic Of': 'Philippines',
    
    # Q
    'Qatar, State Of': 'Qatar',
    
    # R
    'Republic Of Zimbabwe': 'Zimbabwe',
    
    # S
    'Saint Kitts & Nevis': 'Saint Kitts And Nevis',
    'Saint Vincent & The Grenadines': 'Saint Vincent And The Grenadines',
    'Saint Vincent And The Grenadi': 'Saint Vincent And The Grenadines',
    'Saint Vincent The Grenadi': 'Saint Vincent And The Grenadines',
    'San Marino, Republic Of': 'San Marino',
    'Saudi Arabia, Kingdom Of': 'Saudi Arabia',
    'Senegal, Republic Of': 'Senegal',
    'Serbia, Republic Of': 'Serbia',
    'Serbia.': 'Serbia',
    'Seychells': 'Seychelles',
    'Singapore, Republic Of Slovak Republic': 'Singapore',  # Corruption fix
    'Slovakia(Slovak Republic)': 'Slovakia',
    'Soloman Islands': 'Solomon Islands',
    'Somalia, Federal Republic Of': 'Somalia',
    'South Africa- Zuid Afrika': 'South Africa',
    'South Africa, Republic Of': 'South Africa',
    'South Africa-Zuid Afrika': 'South Africa',
    'South Sudan, Republic Of': 'South Sudan',
    
    # T
    'Taiwan': 'Taiwan',
    'Taiwan (Republic Of China)': 'Taiwan',
    'Taiwan Province Of China': 'Taiwan',
    'Tajikistan, Republic Of': 'Tajikistan',
    'Thailand, Kingdom Of': 'Thailand',
    'Timor-Leste, Democratic Republic Of': 'Timor-Leste',
    'Trinidad & Tobago': 'Trinidad And Tobago',
    'Tunisia, Republic Of': 'Tunisia',
    'Turkey, Republic Of': 'Turkey',
    
    # U
    'United States Of America': 'United States',
    'Uruguay, Oriental Republic Of': 'Uruguay',
    'Uzbekistan, Republic Of': 'Uzbekistan',
    
    # V
    'Vanuatu, Republic Of': 'Vanuatu',
    'Vatican City, State Of': 'Vatican City',
    'Venezuela, Bolivarian Republic Of': 'Venezuela',
    
    # Y
    'Yemen (Yemen Arab Rep.)': 'Yemen',
    'Yemen (Yemen Arab Republic)': 'Yemen',
    'Yemen, Republic Of': 'Yemen',
    
    # Z
    'Zambia (Northern Rhodesia)': 'Zambia',
    'Zambia (Northernrhodesia)': 'Zambia',
    'Zambia(Northern Rhodesia)': 'Zambia',
    'Zambia, Republic Of': 'Zambia',
    'Zimbabwe, Republic Of': 'Zimbabwe'
}

print("Applying exact corrections...")
df['Country'] = df['Country'].replace(exact_corrections)

# 2. Pattern-based cleaning for remaining variations
def clean_country_name(name):
    name = str(name).strip()
    
    # Remove common suffixes like ", Republic Of", ", Kingdom Of", etc.
    suffixes_to_remove = [
        r',\s*Republic Of$',
        r',\s*Kingdom Of$',
        r',\s*State Of$',
        r',\s*Principality Of$',
        r',\s*Sultanate Of$',
        r',\s*Commonwealth Of$',
        r',\s*Federative Republic Of$',
        r',\s*People\'S Republic Of$',
        r',\s*Islamic Republic Of$',
        r',\s*Democratic Republic Of$',
        r',\s*Federal Republic Of$',
        r',\s*Federal Democratic Republic Of$',
        r',\s*Cooperative Republic Of$',
        r',\s*Bolivarian Republic Of$',
        r',\s*Oriental Republic Of$',
        r',\s*Union Of$',
        r',\s*Republic Of The$',
        r',\s*Federated States Of$',
        r',\s*Hashemite Kingdom Of$',
        r',\s*Arab Republic Of$',
        r',\s*State Of$',
        r',\s*Democratic People\'S Republic Of$',
        r'\s*\(.*\)$',  # Remove anything in parentheses at the end
    ]
    
    for pattern in suffixes_to_remove:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

print("Applying pattern-based cleaning...")
df['Country'] = df['Country'].apply(clean_country_name)

# 3. Final exact match for any remaining edge cases
final_corrections = {
    'Antigua And Barbuda': 'Antigua And Barbuda',
    'Bosnia And Herzegovina': 'Bosnia And Herzegovina',
    'Brunei Darussalam': 'Brunei Darussalam',
    'Cote D\'Ivoire': 'Cote D\'Ivoire',
    'Democratic Republic Of Congo': 'Democratic Republic Of Congo',
    'Lao People\'S Democratic Republic': 'Laos',
    'The Gambia': 'Gambia',
    'Korea, Democratic People\'S Republic Of': 'North Korea',
    'Korea, Republic Of': 'South Korea',
    'Viet Nam': 'Vietnam',
    'Tanzania, United Republic Of': 'Tanzania',
    'Iran, Islamic Republic Of': 'Iran',
    'Kosovo': 'Kosovo',
    'Congo, Republic Of The': 'Republic Of Congo'
}

df['Country'] = df['Country'].replace(final_corrections)

# Remove the "Country" header row if it exists
df = df[df['Country'] != 'Country']

print(f"\nAfter normalization: {df['Country'].nunique()} unique countries")

# Save the clean dataset
df.to_csv(output_path, index=False)

print(f"\nNormalized data saved to: {output_path}")

# Show the Top 20 countries to prove it worked
print("\nTop 20 Countries (Normalized):")
top_20 = df.groupby('Country')['Arrivals'].sum().sort_values(ascending=False).head(20)
print(top_20)

# Show all unique countries for verification
print(f"\nAll {df['Country'].nunique()} unique countries:")
all_countries = sorted(df['Country'].unique())
for country in all_countries:
    print(f"  - {country}")