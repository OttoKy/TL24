from datasets import  Dataset




def format_instruction(teksti: str, tiivistys: str, eos_token = "<|loppu|>"):
    return f"""<|alku|> Olet tekoälyavustaja. Tiivistä annettu teksti mahdollisimman kattavasti sekä tarkasti.
<|ihminen|> Tässä tiivistettävä teksti:
{teksti.strip()}

<|avustaja|> Vastauksesi:
{tiivistys}{eos_token}
""".strip()

def generate_instruction_dataset(data_point):

    return {
        "artikkeli": data_point["artikkeli"],
        "tiivistys": data_point["tiivistys"],
        "text": format_instruction(data_point["artikkeli"],data_point["tiivistys"])
    }

def process_dataset(data: Dataset):
    columns_to_remove = ['id', 'Unnamed: 0']
    existing_columns = [col for col in columns_to_remove if col in data.column_names]
    return (
        data.shuffle(seed=42)
        .map(generate_instruction_dataset)
        .remove_columns(existing_columns)
    )