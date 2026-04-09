import numpy as np

class ProcrustesAligner:
    @staticmethod
    def align(x: np.ndarray, y: np.ndarray, center: bool = True):
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
        x0 = x - x.mean(axis=0) if center else x
        y0 = y - y.mean(axis=0) if center else y
        u, _, vt = np.linalg.svd(y0.T @ x0, full_matrices=False)
        q = u @ vt
        if np.linalg.det(q) < 0:
            u[:, -1] *= -1
            q = u @ vt
        err = float(np.linalg.norm(y0 - x0 @ q, ord="fro"))
        return q, err

    @staticmethod
    def verify_orthogonality(q: np.ndarray, tol: float = 1e-6):
        ortho = float(np.linalg.norm((q.T @ q) - np.eye(q.shape[0]), ord="fro"))
        det = float(np.linalg.det(q))
        return {"orthogonality_error": ortho, "determinant": det, "is_proper_rotation": ortho < tol and abs(det - 1.0) < tol}
