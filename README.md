# dejavu

> Non ripeto le cose due volte.

**Prompt caching, spiegato.** Perché un LLM rilegge tutto il tuo prompt a ogni domanda — e come smettere di pagarlo ogni volta.

---

## L'idea

Immagina di assumere uno stagista e di dargli un contratto di 50 pagine. Gli fai una domanda e lui, prima di risponderti, si rilegge tutte e 50 le pagine da capo. Gli fai un'altra domanda? Di nuovo, da pagina uno.

Un large language model fa esattamente così. Prima di scrivere una sola parola deve "leggersi" tutto il tuo prompt e costruirsi in testa chi si collega a cosa e dove guardare. Questa fase si chiama **prefill**, ed è la parte lenta e cara — succede prima ancora che parta la risposta.

Il **prompt caching** è dire allo stagista: tieni il contratto aperto, con gli appunti già fatti. Alla domanda dopo non rileggi niente, parti da lì.

Il modello salva quel lavoro di lettura. Tu gli rimandi lo stesso documento ma con una domanda diversa, e lui non rilegge le 50 pagine: recupera gli appunti ed elabora solo la domanda nuova in fondo. Meno attesa, meno costo.

## La regola unica

> Quello che **non cambia** va all'**inizio**. La domanda che **cambia** va alla **fine**.

Il modello confronta il prompt parola per parola dall'inizio e si ferma al primo pezzo diverso. Se metti la domanda davanti, butti via la cache a ogni giro.

## Il conto vero

Chatbot con system prompt + documento = **20.000 token** fissi. **100 domande** nella stessa sessione. Prezzo `gpt-5.4`: input \$2.50 / 1M token, cached \$0.25 / 1M (un decimo).

| | Calcolo | Costo |
|---|---|---|
| **Senza caching** | 100 × 20k = 2M token × \$2.50 | **\$5.00** |
| **Con caching** | 1× pieno + 99× a 1/10 | **~\$0.55** |

Stesso identico chatbot, stesse identiche risposte. **9× meno** — l'unica differenza è aver messo la roba che non cambia all'inizio.

## Due cose da sapere leggendo un pricing

- **L'output costa più dell'input.** Su `gpt-5.4`: \$15 output vs \$2.50 input — 6× tanto. Scrivere costa più che leggere.
- **Short vs long context.** Superi una certa lunghezza → passi alla fascia lunga e l'input raddoppia. Altro motivo per non sprecare token.

## Demo

```bash
python demo.py
```

Manda lo stesso documento N volte, una con la domanda in fondo (cache hit) e una con la domanda in testa (cache busted), e stampa costo + latenza a confronto.

---

Parte della serie: [`goodboy`](.) (RLHF) · [`glassbox`](.) (interpretability) · **`dejavu`** (prompt caching)
