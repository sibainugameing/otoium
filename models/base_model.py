"""
音響モデルの基底クラス

学習済みモデルを統合する際はこのインターフェースを実装する。
現在はルールベース推定のみを使用し、モデルは将来的に追加する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from analysis.feature_extractor import AcousticFeatures
from analysis.boundary_estimator import PhonemeEstimation


class BaseBoundaryModel(ABC):
    """
    音素境界推定モデルの抽象基底クラス。

    将来的に以下のモデルを実装する想定:
    - wav2vec2 ベースのファインチューニング
    - MFC C + CRF による境界推定
    - 熟練者の oto.ini を教師データにした回帰モデル
    """

    @abstractmethod
    def predict(
        self, features: AcousticFeatures
    ) -> tuple[PhonemeEstimation, float]:
        """
        音響特徴量から境界推定を行い、(推定結果, 信頼度) を返す。
        """
        ...

    @abstractmethod
    def load(self, model_path: Path) -> None:
        """学習済みモデルを読み込む"""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """モデルが読み込み済みかどうか"""
        ...


class RuleBasedModel(BaseBoundaryModel):
    """
    ルールベースのフォールバックモデル（常に利用可能）。
    BoundaryEstimator をラップして同一インターフェースを提供する。
    """

    def __init__(self):
        from analysis.boundary_estimator import BoundaryEstimator
        self._estimator = BoundaryEstimator()
        self._loaded = True

    def predict(
        self, features: AcousticFeatures
    ) -> tuple[PhonemeEstimation, float]:
        raise NotImplementedError(
            "RuleBasedModel は predict() を直接呼ばない。"
            "BoundaryEstimator.estimate_cv/vcv() を使用してください。"
        )

    def load(self, model_path: Path) -> None:
        pass  # ルールベースはモデルファイルを持たない

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def load_model(model_path: Optional[Path] = None) -> BaseBoundaryModel:
    """
    利用可能なモデルを読み込んで返す。
    モデルが指定されていない場合はルールベースを返す。
    """
    if model_path is None:
        return RuleBasedModel()

    # 将来: モデルの種類を拡張子や設定ファイルで判定して読み込む
    raise NotImplementedError(f"外部モデルはまだ実装されていません: {model_path}")
