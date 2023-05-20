# Bachelor-Project
Utnytting av symmetri til å kunstig øke mengden treningsdata for klassifisering av kalorimeterbilder fra ATLAS (HVL) 

ATLAS-detektoren måler utfallet av proton-proton-kollisjoner produsert av Large Hadron Collider (LHC) på CERN. Data-settet som samles inn ved hjelp av ATLAS-detektoren analyseres for å bedre forstå fysikken på mikroskala. Et spesielt fokus er på søk etter fenomener som ikke er konsistent med eksisterende modeller – ofte omtalt som «ny fysikk». Vi vet at slik ny fysikk finnes da astrofysiske observasjoner, f.eks. eksistensen av mørk materie og overskuddet av materie relativt til antimaterie i universet, ikke kan forklares ut fra eksisterende modeller for partikkelfysikk. 

I kollisjoner mellom sub-atomære partikler, som for eksempel protoner, dannes det nye partikler. Ved så store energier som er tilgjengelig i LHC kan alle kjente partikkeltyper dannes. Selve kollisjonsprosessen er stokastisk, noe som innebærer at det er umulig å forutsi hva som blir utfallet av den enkelte kollisjon. Kvantemekaniske beregninger gir oss imidlertid muligheten til å forutsi sannsynligheten for ulike utfall. 

Den konvensjonelle metoden for innsamling og analyse av data fra kollisjonseksperimenter slik som ATLAS-eksperimentet er: 

Trigger og dataregistrering 

En kombinasjon av dedikert maskinvare og spesialisert programvare gjør et raskt utvalg av hvilke kollisjoner registrert av detektoren som skal leses ut og lagres. Dette steget er nødvendig da kollisjonsraten er mye større enn det som er mulig å håndtere både med tanke på utlesningshastighet og datalagringskapasitet.  

Rekonstruksjon 

Data som leses ut fra detektoren består av elektriske signaler som forteller hvor partikler som kom ut av proton-proton-kollisjonen vekselvirket med detektoren. Ved hjelp av spesialisert programvare blir rådata satt sammen og tolket slik at man får en liste over hvilke partikler som kom ut fra kollisjonen. Settet av rekonstruerte partikler inneholder informasjon om både partikkeltype, og om kinematiske variabler som energi og bevegelsesretning. 

Analyse 

Basert på settet av rekonstruerte partikler analyseres kollisjons-dataene for å tolke hva som skjedde i kollisjonen. Et viktig steg i analysen er å klassifisere de ulike kollisjonene basert på hvilke partikler som ble dannet, og så sammenligne dette med teoretiske beregninger for å avgjøre om fordelingen av utfall er konsistent med eksisterende modeller.  

Målet ved dette prosjektet er å gjøre et bidrag til forsøk på en ny analyse-strategi som slår sammen rekonstruksjon- og analysesteget. Spesifikt ønsker vi å undersøke om det er mulig å forbedre sensitiviteten i analysen ved å trene opp en maskinlæringsalgoritme til å klassifisere kollisjonsdataene basert på rådata (eller kun delvis rekonstruerte data) fra hele eller deler av detektoren. 

ATLAS-detektoren kan i denne sammenhengen litt forenklet betraktes som et sylindrisk digital-kamera som fotograferer kollisjonene. Videre vil symmetrien i kollisjonene gjenspeiles i en høyre-venstre-symmetri. Ved å «brette ut» sylinderen får vi da rektangulære bilder som kan analyseres ved hjelp av maskinlæringsalgoritmer for bildegjenkjenning.  

I prosjektet vil det bli brukt simulerte kollisjons-data som gjør at den riktige klassifikasjonen er kjent. Dette gjør datasettet velegnet til bruk for veiledet maskinlæring. Videre gir det muligheten til å tilpasse datasettet underveis ved å produsere mer av spesifikke prosesser dersom arbeidet viser behov for det. 


Mål for prosjektet: 

- Tilpasse første pooling-lag av et CNN til å ta hensyn til sylindersymmetrien, og høyre-venstresymmetrien til detektor-bildene. 

- Undersøke om, og hvordan, symmetri-egenskapene til detektorbildene kan utnyttes til «data augmentation» uten å innføre bias i klassifikasjonen. 

# Mappe struktur

├───data
│       BH_n4_M10_res50_15000_events.h5
│       PP13-Sphaleron-THR9-FRZ15-NB0-NSUBPALL_res50_15000_events.h5
│
├───Extra
│       heatmap.ipynb
│       visual.ipynb
│
├───methods
│   │   .gitignore
│   │   directory_structure.txt
│   │   fold.txt
│   │   folder.txt
│   │   README.md
│   │
│   ├───__pycache__
│   │       dataloader.cpython-39.pyc
│   │       nnmodel.cpython-39.pyc
│   │       plotCreator.cpython-39.pyc
│   │       train.cpython-39.pyc
│   │       trainer.cpython-39.pyc
│   │       trainerDataAug.cpython-39.pyc
│   │       trainerRet.cpython-39.pyc
│   │
│   ├───dataloader.py
│   ├───nnmodel.py
│   ├───plotCreator.py
│   ├───train.py
│   ├───trainer.py
│   └───trainerRet.py
│
└───notebooks
    ├───fast.ai
    │       firstModell.ipynb
    │
    └───pyTorch
        ├───Trained_models
        │       FInal
        │       model_checkpoint.pth
        │       model_checkpoint_aug.pth
        │       model_checkpoint_augResSymm.pth
        │       model_checkpoint_aug_SymmNet.pth
        │       model_checkpoint_aug_VGGNet.pth
        │       model_checkpoint_aug_VGGNet2.pth
        │       model_checkpoint_aug_VGGNet3.pth
        │       model_checkpoint_aug_VGGNet4.pth
        │       model_checkpoint_aug_VGGNet5.pth
        │       model_checkpoint_aug_VGGNet6.pth
        │       model_checkpoint_aug_VGGNet7.pth
        │       model_ConvModel.pth
        │       model_ConvModel1.pth
        │       VGGNETLAST.pth
        │
        ├───Combined_Data_Augm.ipynb
        ├───ConvoCompare.ipynb
        ├───ConvoModRuns copy.ipynb
        ├───ConvoRuns.ipynb
        ├───DataAugm_CNNMod.ipynb
        ├───DataAugm_CNNSimple.ipynb
        ├───FinalCNNMod.ipynb
        ├───FinalCNNModRoll.ipynb
        ├───FinalCNNSimple.ipynb
        ├───FinalVGGNet.ipynb
        ├───ResNet.ipynb
        ├───ResnetSymUpd.ipynb
        └───VanligCNN.ipynb
        
# Miljø oppsett
Anaconda env deretter
Python
Pytorch
Numpy
conda
matplotlib
fastai
sys
pathlib

# In case commits link to another user
Open terminal in vscode (despite what kernel you used)
git config user.email "Your email"
git config user.email
