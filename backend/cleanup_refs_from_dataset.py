import sys
import os
sys.path.append(os.getcwd())

from database import SessionLocal, DatasetImage, ReferenceImage

def cleanup():
    session = SessionLocal()
    try:
        # Get all reference paths
        refs = session.query(ReferenceImage).all()
        ref_paths = [r.path for r in refs]
        
        if not ref_paths:
            print("No reference images found.")
            return

        print(f"Found {len(ref_paths)} reference images.")

        # Find dataset images that match reference paths
        query = session.query(DatasetImage).filter(DatasetImage.original_path.in_(ref_paths))
        count = query.count()
        
        if count > 0:
            deleted = query.delete(synchronize_session=False)
            session.commit()
            print(f"Successfully removed {deleted} duplicate reference images from dataset.")
        else:
            print("No duplicates found in dataset.")
            
    except Exception as e:
        print(f"Error: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    cleanup()
