def get_files_in_folder(inputDir, pattern):
    import os
    import glob
    filePattern = os.path.join(inputDir,  pattern)
    matchingFiles = glob.glob(filePattern)
    fileNames = [os.path.basename(file) for file in matchingFiles]
    return fileNames

