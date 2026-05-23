import os


def extract_citi_names(directory):
    city_company_pairs = set()

    for fname in os.listdir(directory):
        if fname.lower().endswith('.png'):
            name_without_ext = fname[:-4]
            parts = name_without_ext.rsplit('-', 1)

            if len(parts) != 2:
                print(f"⚠️  Filename format does not match 'city-company.png': {fname}")
                continue

            city_raw, company = parts[0].strip(), parts[1].strip()
            city = ''.join(char for char in city_raw if char.isalpha() or char in [' ', '-'])
            city = city.strip()

            if city == "Sanya":
                city = "Sanya, China"
            elif city == "Bay Area":
                city = "San Francisco"
            elif city == "Silicon Valley":
                city = "San Francisco"

            if city and company:
                city_company_pairs.add((city, company))
            else:
                print(f"⚠️  Cannot parse city or company: {fname}")

    city_counts = {}
    for city, _ in city_company_pairs:
        city_counts[city] = city_counts.get(city, 0) + 1
    return city_counts