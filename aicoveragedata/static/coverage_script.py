from data_utils  import load_dataset
import matplotlib as plt
df = load_dataset()



# fixed layout for each sections title (just makes it look better visually)
def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")




#Overall Data Coverage
observations, variables = df.shape
section("Dataset Dimensions")
print(f"Observations: {observations}".center(35))
print(f"Variables: {variables}".center(35))



#Amount of unique determinants
start_year = df["year"].min()
end_year = df["year"].max()
individual_company_num = df["company_id"].nunique() 
individual_industries = df["industry"].nunique() 
individual_countries = df["country"].nunique() 

section("Coverage")
print(f"Years Covered: {start_year} - {end_year} ".center(20)) 
print(f"Individual Companies: {individual_company_num}".center(20))
print(f"Individual Industries: {individual_industries}".center(20))
print(f"Individual Countries: {individual_countries}".center(20))

# Unique Industries
industry_list = df["industry"].unique().tolist()
section("Industries Covered")

for industry in industry_list:
    print(f"- {industry}".center(30))

# Unique Countries
country_list = df["country"].unique().tolist()
section("Countries Covered")

for country in country_list:
    print(f"- {country}".center(30))






