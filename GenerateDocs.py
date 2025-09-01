"""!
@file
PyPO docs generator.

For this script to work properly, you should have installed the docs prerequisites.
"""

import os
import shutil
import traceback

def GenerateDocs():
    docPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "docs")
    demoPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join("etc", "demos"))
    doxyPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "doxy")
    try:
        try:
            shutil.rmtree(docPath)
        except Exception:
            print(traceback.format_exc())

        # Convert demos to html format for inclusion in the documentation.
        for (dirpath, dirnames, filenames) in os.walk(demoPath):
            destPath = os.path.join(docPath, "demos")
            os.makedirs(destPath)
            for file in filenames:
                if file.split('.')[1] != "ipynb":
                    continue

                _path = os.path.join(demoPath, file)
                htmlPath = os.path.join(demoPath, f"{file.split('.')[0]}.html")
                html_destPath = os.path.join(destPath, f"{file.split('.')[0]}.html")

                os.system(f"jupyter nbconvert --to html --template lab --theme dark {_path}")
                os.rename(htmlPath, html_destPath)
            
            break

        os.system(f"doxygen {os.path.join(doxyPath, 'Doxyfile')}")
        
        filelistPath = os.path.join(docPath, "files.html")
        with open(filelistPath, 'r') as file :
            filedata = file.read()
            filedata = filedata.replace('File List', 'Full Software Documentation')
            filedata = filedata.replace("Here is a list of all documented files with brief descriptions:",
                                        "Here is a list containing the full software documentation. The structure of this page reflects the source code hierarchy.")
        with open(filelistPath, 'w') as file:
            file.write(filedata)
        
    except Exception:
        print(traceback.format_exc())
    
if __name__ == "__main__":
    GenerateDocs()
