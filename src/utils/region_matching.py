import pandas as pd
import country_converter as coco

from src.config import (
    INPUT_DIR
)

# -------------------------------------------------------------------
# Basin and Region Mapping
# -------------------------------------------------------------------
def create_un_regions_csv():
    """Generates the ISO3 to UN Region mapping required for model training."""
    out_path = INPUT_DIR / "model_input_dataset" / "un_regions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if out_path.exists(): return
    
    # Use the ISO3_LIST from config
    from src.config import ISO3_LIST
    df = pd.DataFrame({"iso3": ISO3_LIST})
    df['region'] = coco.convert(names=df['iso3'], to='UNregion')
    
    df.to_csv(out_path, index=False)
    print(f"Created regional mapping at {out_path}")

def add_basin_information(df):
    """
    Appends the UN Region, Continent, and Cyclone Basin based on the ISO3 country code.
    """
    if "iso3" not in df.columns:
        raise ValueError("Dataframe must contain an 'iso3' column.")

    unique_iso3 = df[["iso3"]].drop_duplicates().reset_index(drop=True)
    
    unique_iso3['continent'] = coco.convert(names=unique_iso3['iso3'], to='continent')
    unique_iso3['region'] = coco.convert(names=unique_iso3['iso3'], to='UNregion')

    un_region_to_basin = {
        "Caribbean": "North Atlantic",
        "Central America": "North Atlantic",
        "South America": "North Atlantic",
        "Northern America": "North Atlantic",
        "Australia and New Zealand": "Australian Region",
        "Southern Asia": "North Indian",
        "Eastern Asia": "Western Pacific",
        "South-eastern Asia": "Western Pacific",
        "Western Asia": "North Indian",
        "Eastern Africa": "South-West Indian",
        "Southern Africa": "South-West Indian",
        "Melanesia": "South Pacific",
        "Micronesia": "Western Pacific",
        "Polynesia": "South Pacific",
        "Southern Europe": "Europe",
    }
    
    unique_iso3["cyclone_basin"] = unique_iso3["region"].map(un_region_to_basin)
    
    return df.merge(unique_iso3, on="iso3", how="left")

def create_basin_dataset():
    """ Generates a dataset with UN region and Cyclone Basin information """
    df = create_un_regions_csv()
    return add_basin_information(df)

if __name__ == "__main__":
    create_basin_dataset()
