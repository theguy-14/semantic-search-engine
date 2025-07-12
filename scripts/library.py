import json
import os

def update_library_entry(file_name, doc_id, library_name):
    try:
        metadata_path = "./data/pdf_metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        doc_entry = metadata[doc_id]
        doc_entry.setdefault("library", [])

        if library_name not in doc_entry["library"]:
            doc_entry["library"].append(library_name)

        # Write back to file
        # with open(metadata_path, "w", encoding="utf-8") as f:
        #     json.dump(metadata, f, indent=2, ensure_ascii=False)

        return {
            "status": "success",
            "message": f"{file_name} added to {library_name}",
            "doc_id": doc_id,
            "current_libraries": doc_entry["library"]
        }

    except KeyError as e:
        return {"status": "error", "message": f"Missing or invalid key: {str(e)}"}
    except FileNotFoundError:
        return {"status": "error", "message": "metadata.json not found"}
    except json.JSONDecodeError:
        return {"status": "error", "message": "metadata.json is corrupted"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

def get_docs_in_library(library_name):
    try:
        metadata_path = "./data/pdf_metadata.json"

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        doc_map = {
            doc_id: data.get("title", "Untitled")
            for doc_id, data in metadata.items()
            if "library" in data and library_name in data["library"]
        }

        return {
            "status": "success",
            "library": library_name,
            "documents": doc_map
        }

    except FileNotFoundError:
        return {"status": "error", "message": "metadata.json not found"}
    except json.JSONDecodeError:
        return {"status": "error", "message": "metadata.json is corrupted"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}
