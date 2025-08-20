import re
import pandas as pd
import numpy as np
from urllib.parse import urlparse, parse_qs
import tldextract
from sklearn.base import BaseEstimator, TransformerMixin

class UrlStatFeatures(BaseEstimator, TransformerMixin):
    def __init__(self): pass
    def fit(self, X, y=None): return self

    def _ensure_series(self, X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 1:
                return X.iloc[:, 0].astype(str)
            raise ValueError("UrlStatFeatures espera 1 columna con la URL.")
        elif isinstance(X, pd.Series):
            return X.astype(str)
        elif isinstance(X, (list, tuple, np.ndarray)):
            return pd.Series(X).astype(str)
        else:
            return pd.Series([str(X)])

    def transform(self, X):
        s = self._ensure_series(X)

        def feats(url):
            u = urlparse(url)
            ext = tldextract.extract(url)
            q = parse_qs(u.query)
            host = u.netloc or ""
            path = u.path or ""
            query = u.query or ""
            subd = ext.subdomain or ""
            suffix = ext.suffix or ""
            domain = ext.domain or ""

            num_digits = sum(c.isdigit() for c in url)
            num_letters = sum(c.isalpha() for c in url)
            num_dots = url.count(".")
            num_hyphens = url.count("-")
            has_ip = bool(re.search(r"(?:\d{1,3}\.){3}\d{1,3}", host))
            has_at = "@" in url
            has_https = url.lower().startswith("https://")

            return {
                "len_url": len(url),
                "num_digits": num_digits,
                "num_letters": num_letters,
                "num_dots": num_dots,
                "num_hyphens": num_hyphens,
                "has_ip": int(has_ip),
                "has_at": int(has_at),
                "has_https": int(has_https),
                "len_host": len(host),
                "len_path": len(path),
                "len_query": len(query),
                "num_params": len(q),
                "len_subdomain": len(subd),
                "len_domain": len(domain),
                "len_suffix": len(suffix),
            }

        rows = [feats(u) for u in s.tolist()]
        return pd.DataFrame(rows)

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "len_url","num_digits","num_letters","num_dots","num_hyphens",
            "has_ip","has_at","has_https","len_host","len_path","len_query",
            "num_params","len_subdomain","len_domain","len_suffix"
        ])
