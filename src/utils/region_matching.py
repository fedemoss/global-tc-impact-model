import os
import pandas as pd
import country_converter as coco
from src.config import INPUT_DIR

# -------------------------------------------------------------------
# Basin and Region Mapping
# -------------------------------------------------------------------
def create_un_regions_csv():
    """Generates the ISO3 to UN Region and Continent mapping."""
    out_path = INPUT_DIR / "model_input_dataset" / "un_regions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if out_path.exists(): 
        return pd.read_csv(out_path)
    
    from src.config import ISO3_LIST
    df = pd.DataFrame({"iso3": ISO3_LIST})
    df['region'] = coco.convert(names=df['iso3'], to='UNregion')
    df['continent'] = coco.convert(names=df['iso3'], to='continent')
    
    df.to_csv(out_path, index=False)
    return df

def add_basin_information(df):
    """
    Appends the Cyclone Basin based on the UN Region.
    """
    if "region" not in df.columns:
        raise ValueError("Dataframe must contain a 'region' column.")

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
    
    df["cyclone_basin"] = df["region"].map(un_region_to_basin)
    return df

def create_basin_dataset():
    """Generates a dataset with UN region, Continent, and Cyclone Basin information."""
    df = create_un_regions_csv()
    return add_basin_information(df)

def create_region_dataset():
    """Main pipeline for region dataset creation."""
    df = create_basin_dataset()
    out_dir = INPUT_DIR / "model_input_dataset"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "un_regions.csv"
    
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    create_region_dataset()