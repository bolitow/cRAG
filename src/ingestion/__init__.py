# src/ingestion/__init__.py
"""
Module d'ingestion de documents pour le système CRAG.

Ce module fournit une infrastructure complète pour :
- Parser différents formats de documents (PDF, Markdown, Excel, FAQ)
- Extraire et enrichir les métadonnées
- Préprocesser les documents pour le domaine de la cybersécurité
"""

from .document_parser import DocumentParser
from .parsers.base_parser import ParsedDocument, ParsingError
from .preprocessors.security_preprocessor import SecurityPreprocessor
from .preprocessors.metadata_extractor import MetadataExtractor

__all__ = [
    'DocumentParser',
    'ParsedDocument',
    'ParsingError',
    'SecurityPreprocessor',
    'MetadataExtractor'
]

# Version du module
__version__ = '1.0.0'