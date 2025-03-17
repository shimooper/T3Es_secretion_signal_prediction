import pandas as pd

from .process_raw_data import RAW_DATA_PATH

DB_PATH = RAW_DATA_PATH / 'T3Edb_summary.csv'


def validate_xanthomonas(logger, t3e_records):
    t3e_xanthomonas_records = [record for record in t3e_records if
                               'Xanthomonas' in record.description or 'Xanthomonas' in record.id]

    db_df = pd.read_csv(DB_PATH, encoding='utf-8', encoding_errors='replace')
    db_xanthomonas_df = db_df[db_df['species'].str.startswith('Xanthomonas')]
    db_xanthomonas_ids = list(db_xanthomonas_df['rec_id'])
    logger.info(
        f'There are {len(t3e_xanthomonas_records)} Xanthomonas in T3Es fasta and {len(db_xanthomonas_df)} in the database.')

    fasta_xanthomonas_ids = []
    for record in t3e_xanthomonas_records:
        match = re.search(r'^(.*?)_Xanthomonas', record.description)
        if match is None:
            logger.error(f'Xanthomonas record {record.id} from fasta does not have a valid record id.')
            continue

        rec_id = match.group(1)
        fasta_xanthomonas_ids.append(rec_id)
        fasta_xanthomonas_ids.append(record.id)
        if rec_id not in db_xanthomonas_ids and record.id not in db_xanthomonas_ids:
            logger.error(f'Xanthomonas record {record.id} from fasta is not in the database.')

    for record_id in db_xanthomonas_ids:
        if record_id not in fasta_xanthomonas_ids:
            logger.error(f'Xanthomonas record {record_id} from db is not in the fasta.')