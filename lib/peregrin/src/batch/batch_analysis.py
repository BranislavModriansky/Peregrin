import pandas as pd


class ComputeBatch:
    
    def __init__(self): ...

    def compute(
        self, 
        dir: str, 
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute statistics for all data files in the specified directory.

        Parameters
        ----------
        dir : str
            The path to the directory containing the data files.

        kwargs : dict
            Additional keyword arguments to pass to the `stats` method.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the computed statistics for all data files.
        """


        self.dir = dir
        self.kwargs = kwargs
