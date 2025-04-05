import requests
import os
import logging
from src.data.check_structure import check_existing_file, check_existing_folder


def import_raw_data(raw_data_relative_path, 
                    filenames,
                    bucket_folder_url):
    '''import filenames from bucket_folder_url in raw_data_relative_path'''

                
def main(raw_data_relative_path="./data/raw", 
        filenames = ["genome-scores.csv", "genome-tags.csv", "links.csv", 
                    "movies.csv", "ratings.csv", "README.txt", "tags.csv"],
        bucket_folder_url= "https://mlops-project-db.s3.eu-west-1.amazonaws.com/movie_recommandation/"          
        ):
    """ Upload data from AWS s3 in ./data/raw
    """
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    logger = logging.getLogger(__name__)
    logger.info('making raw data set')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
