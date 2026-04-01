-- PostgreSQL schema for migrated SQLite table `prodotti`

CREATE TABLE IF NOT EXISTS prodotti (
    id TEXT PRIMARY KEY,
    nome TEXT,
    fornitore TEXT,
    codice TEXT,
    prezzo DOUBLE PRECISION,
    cliente TEXT,
    offerta_num TEXT,
    specifiche TEXT,
    descrizione TEXT,
    data_offerta TEXT
);

CREATE INDEX IF NOT EXISTS idx_prodotti_codice ON prodotti(codice);
CREATE INDEX IF NOT EXISTS idx_prodotti_cliente ON prodotti(cliente);
CREATE INDEX IF NOT EXISTS idx_prodotti_offerta_num ON prodotti(offerta_num);
CREATE INDEX IF NOT EXISTS idx_prodotti_data_offerta ON prodotti(data_offerta);
