import logging
from deepsearcher.vector_db.milvus import Milvus
from deepsearcher.configuration import Configuration, init_config
from deepsearcher import configuration
from pymilvus import Collection

# Set up logging for better output management
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded list of collections to always preserve (modify as needed)
HARDCODED_PRESERVE = []  # Example: ['energy_sustainability_strategy', 'vctf_investment']

def main():
    # Initialize configuration and vector database
    config = Configuration()
    init_config(config=config)
    vector_db = configuration.vector_db
    
    # Get all collection names
    all_collections = [col.collection_name for col in vector_db.list_collections()]
    
    # Ask user for action
    action = input("What do you want to do? Please type one of the two options: [delete]/[rename]): ").strip().lower()
    
    if action == "delete":
        delete_type = input("Delete specific collection or all non-preserved? (specific/all): ").strip().lower()
        if delete_type == "specific":
            # Display all available collections
            logger.info("Available collections for deletion:")
            for col in all_collections:
                logger.info(f"- {col}")
            delete_input = input("Enter the collection names to delete, separated by spaces (e.g., 'col1 col2 col3'): ").strip()
            cols_to_delete = delete_input.split()
            valid_cols_to_delete = []
            for col in cols_to_delete:
                if col not in all_collections:
                    logger.warning(f"Collection '{col}' does not exist.")
                elif col in HARDCODED_PRESERVE:
                    logger.info(f"Collection '{col}' is preserved and cannot be deleted.")
                else:
                    valid_cols_to_delete.append(col)
            if valid_cols_to_delete:
                logger.info(f"Collections to delete: {valid_cols_to_delete}")
                confirm = input(f"Are you sure you want to delete these {len(valid_cols_to_delete)} collections? (y/n): ").strip().lower()
                if confirm == 'y':
                    for col in valid_cols_to_delete:
                        try:
                            vector_db.clear_db(collection=col)
                            logger.info(f"Deleted collection: {col}")
                        except Exception as e:
                            logger.error(f"Failed to delete collection {col}: {e}")
                else:
                    logger.info("Deletion aborted.")
            else:
                logger.info("No valid collections to delete.")
        elif delete_type == "all":
            # Ask for additional collections to preserve
            additional_preserves = input("Enter additional collections to preserve (comma-separated), or press Enter: ").strip()
            preserve_set = set(HARDCODED_PRESERVE)
            if additional_preserves:
                preserve_set.update([col.strip() for col in additional_preserves.split(',')])
            collections_to_delete = [col for col in all_collections if col not in preserve_set]
            if collections_to_delete:
                logger.info(f"Collections to delete: {collections_to_delete}")
                confirm = input(f"Are you sure you want to delete these {len(collections_to_delete)} collections? (y/n): ").strip().lower()
                if confirm == 'y':
                    for col in collections_to_delete:
                        try:
                            vector_db.clear_db(collection=col)
                            logger.info(f"Deleted collection: {col}")
                        except Exception as e:
                            logger.error(f"Failed to delete collection {col}: {e}")
                else:
                    logger.info("Deletion aborted.")
            else:
                logger.info("No collections to delete.")
        else:
            logger.error("Invalid choice. Please choose 'specific' or 'all'.")
    
    elif action == "rename":
        # Display all available collections
        logger.info("Available collections for renaming:")
        for col in all_collections:
            logger.info(f"- {col}")
        
        # Ask for multiple rename pairs
        rename_input = input("Enter the renames in the format 'old_name1:new_name1 old_name2:new_name2 ...' (e.g., 'old1:new1 old2:new2'): ").strip()
        pairs = rename_input.split()
        renames_to_perform = []
        for pair in pairs:
            try:
                old_name, new_name = pair.split(":")
                if old_name == new_name:
                    logger.info(f"Old and new names are the same for '{old_name}', skipping.")
                    continue
                if old_name not in all_collections:
                    logger.error(f"Collection '{old_name}' does not exist.")
                    continue
                if new_name in all_collections:
                    logger.error(f"Collection '{new_name}' already exists.")
                    continue
                renames_to_perform.append((old_name, new_name))
            except ValueError:
                logger.error(f"Invalid format for pair: {pair}")
        if renames_to_perform:
            logger.info("The following renames will be performed:")
            for old, new in renames_to_perform:
                logger.info(f"- '{old}' to '{new}'")
            confirm = input("Are you sure you want to perform these renames? (y/n): ").strip().lower()
            if confirm == 'y':
                for old, new in renames_to_perform:
                    try:
                        vector_db.rename_collection(old, new)
                        logger.info(f"Renamed collection '{old}' to '{new}'")
                    except Exception as e:
                        logger.error(f"Failed to rename collection '{old}' to '{new}': {e}")
            else:
                logger.info("Rename aborted.")
        else:
            logger.info("No valid renames to perform.")
    
    else:
        logger.error("Invalid action. Please choose 'delete' or 'rename'.")
    
    # List remaining collections and their references
    remaining_collections = vector_db.list_collections()
    if remaining_collections:
        logger.info(f"Remaining collections: {[col.collection_name for col in remaining_collections]}")
        collection_objects = set()
        for col in remaining_collections:
            try:
                collect_obj = Collection(name=col.collection_name)
                collection_objects.add(collect_obj)
            except Exception as e:
                logger.error(f"Failed to access collection {col.collection_name}: {e}")
                continue
        
        embedded_refs = set()
        for col in collection_objects:
            try:
                # Query all documents, retrieving only the "reference" field
                docs = col.query(expr="", output_fields=["reference"], limit=16384)
                for doc in docs:
                    if "reference" in doc:
                        embedded_refs.add(doc["reference"])
            except Exception as e:
                logger.error(f"Failed to query collection {col.name}: {e}")
        
        # Provide summarized feedback instead of raw references
        if embedded_refs:
            logger.info(f"Total unique references across preserved collections: {len(embedded_refs)}")
        else:
            logger.info("No references found in preserved collections.")
    else:
        logger.info("No collections remaining.")

if __name__ == "__main__":
    main()