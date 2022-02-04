import plac
import random
from pathlib import Path
import spacy
from spacy.tokens import Doc
from spacy.training import Example


# new entity label
LABEL = 'ANIMAL'

TRAIN_DATA = [
("Horses are too tall and they pretend to care about your feelings", {
'entities': [(0, 6, 'ANIMAL')]
}),
("Do they bite?", {
'entities': []
}),
("horses are too tall and they pretend to care about your feelings", {
'entities': [(0, 6, 'ANIMAL')]
}),
("horses pretend to care about your feelings", {
'entities': [(0, 6, 'ANIMAL')]
}),
("they pretend to care about your feelings, those horses", {
'entities': [(48, 54, 'ANIMAL')]
}),
("horses?", {
'entities': [(0, 6, 'ANIMAL')]
})
]

predicted = Doc(spacy.vocab.Vocab(),words=['horse','dog','fox','elephant','ghughu'])

token_ref = ['horse','dog','fox','elephant','ghughu']
tags_ref = ["Animal","Animal","Animal","Animal","Animal"]

ex = Example.from_dict(predicted,{"words":token_ref,"tags":tags_ref})

lex = [ex for i in range(10)]


@plac.annotations(
model=("model", "option", "m",
str),
new_model_name=("new_model_name", "option", "nm", str),
output_dir=("output_dir", "option", "o", Path),
n_iter=("n_iter", "option", "n", int))
def main(model=None, new_model_name='animal', output_dir=None, n_iter=20):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
        # Add entity recognizer to model if it's not in the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        #ner = nlp.create_pipe('ner')
        nlp.add_pipe('ner')
        # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')
        ner.add_label(LABEL)
    if model is None:
        print("Model is None")
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.create_optimizer()
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe!='ner']
    with nlp.disable_pipes(*other_pipes):
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text,annotations in TRAIN_DATA:
                #nlp.update([text],[annotations],sgd=optimizer,drop=0.35,losses=losses)
                nlp.update(lex,sgd=optimizer,drop=0.35,losses=losses)
            print(losses)
                
    #test the model
    test_text = "Do you like horses?"
    doc = nlp(test_text)
    print("Entities are")
    for ent in doc.ents:
        print(ent.label_,ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)



if __name__ == '__main__':
    plac.call(main)