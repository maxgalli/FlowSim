import sys
import fastjet as fj
import pythia8
from random import uniform, gauss as normal
from numpy.random import poisson

from math import *
import numpy as np

from generator_args import get_args

args = get_args()

# Initialize Pythia and set up the process
pythia = pythia8.Pythia()
pythia.readString("Random:setSeed = on")
pythia.readString(f"Random:seed = {args.seed}")
pythia.readString("Beams:eCM = 13000.")
if args.ttbar:
    pythia.readString("Top:gg2ttbar = on")
    pythia.readString("Top:qqbar2ttbar = on")
elif args.qcd:
    pythia.readString("HardQCD:all = on")
    pythia.readString("PhaseSpace:pTHatMin = 100.")
elif args.diboson:
    pythia.readString("WeakDoubleBoson:ffbar2WW = on")
elif args.zjets:
    pythia.readString("WeakBosonAndParton:qqbar2gmZg = on")
    pythia.readString("WeakBosonAndParton:qg2gmZq = on")
else:
    print("Please specify a process to generate")

pythia.init()

# Jet clustering parameters
jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)


def addJetTagging(jet):
    if abs(jet.flavour) == 5:
        jet.nSV = poisson(0.8)
        jet.btag = atan(uniform(0, jet.pt() / 2)) / pi * 2.0
        jet.ctag = uniform(0, 1)
        jet.qgl = uniform(0, 1) ** 0.5 + jet.btag / 5.0
    elif abs(jet.flavour) == 4:
        jet.nSV = poisson(0.3)
        jet.btag = uniform(0, 1)
        jet.ctag = atan(uniform(0, jet.pt() / 2)) / pi * 2.0
        jet.qgl = uniform(0, 1) ** 0.75 + jet.btag / 5.0
    else:
        jet.nSV = poisson(0.1)
        jet.btag = 1.0 - atan(uniform(jet.pt() / 2, 0)) / pi * 2.0
        jet.ctag = jet.btag / 2 + uniform(0, 0.5)
        if abs(jet.flavour) == 21:
            jet.qgl = uniform(0, 1) ** 2
        else:
            jet.qgl = uniform(0, 1) ** 0.8
    if (
        len(jet.muonsInJet) > 0
    ):  # if there are muons the tagger will consider it more b-like
        jet.btag = 1.0 - (1.0 - jet.btag) / 2.5
    if jet.nSV > 0:
        jet.btag = 1.0 - (1.0 - jet.btag) / 1.5
        jet.ctag = 1.0 - (1.0 - jet.ctag) / 1.5

    return jet


def addJetRecoKinematics(jet):
    # bias = (
    #     0.95 if jet.flavour == 5 else 1.02
    # )  # add some different reco bias  for B vs non B
    bias = (
        (0.5 + (jet.flavour == 5) * (-0.05)) * exp(-jet.pt() / 15)
        + 1
        + (jet.flavour == 5) * (-0.02)
    )
    width = (
        0.15 + 2 / jet.pt() + (jet.flavour == 5) * 0.02 + abs(jet.eta()) * 0.002
    )  # scale resolution with pt, eta and flavour
    pTresponse = normal(bias, width)
    mResponse = normal(pTresponse, 0.1)
    jet.recoPt = jet.pt() * pTresponse
    jet.recoEta = jet.eta() * normal(1.0, 0.02)  # just smearing for eta and phi
    jet.recoPhi = jet.phi() * normal(1.0, 0.01)
    jet.recoMass = jet.m() * mResponse

    jet.recoNConstituents = poisson(len(jet.pythiaConstituents))
    if jet.recoNConstituents < 0:
        jet.recoNConstituents = 0

    jet.ncharged = int(
        len([x for x in jet.pythiaConstituents if x.charge() != 0])
        * poisson(len(jet.pythiaConstituents))
        / len(jet.pythiaConstituents)
    )
    if jet.ncharged < 0:
        jet.ncharged = 0

    jet.nneutral = jet.recoNConstituents - jet.ncharged
    if jet.nneutral < 0:
        jet.nneutral = 0

    jet.qgl /= (5 + jet.recoNConstituents) / 30
    jet.jetId = (
        sin(jet.chf + jet.cef) + jet.recoNConstituents / 100 + jet.pt() / 250.0
    ) / 2
    jet.jetId = jet.jetId + normal(0, 0.05)
    #    print(jet.pt(), jet.eta(), jet.flavour," -> ", width)
    #    print("      ", jet.pt(), jet.recoPt)
    return jet


# Function to check if a particle is stable (excluding neutrinos)
def is_stable(p):
    return p.isFinal() and p.idAbs() != 12 and p.idAbs() != 14 and p.idAbs() != 16


# Function to create a collection of final muons or electrons
def filter_particles(particles, id_abs, requireFinal=True):
    return [
        p
        for p in particles
        if (not requireFinal or p.isFinal()) and p.idAbs() in id_abs
    ]


# Function to match jets to original particles and determine the jet flavor
def matchAndClean(jets, particles):
    matched_jets = []
    for jet in jets:
        if len(jet.muonsInJet) > 0:
            iso = jet.muonsInJet[0].pT() / jet.pt()
            if iso > 0.8:
                continue  # skip jets that are actually just single leptons
        if len(jet.elesInJet) > 0:
            iso = jet.elesInJet[0].pT() / jet.pt()
            if iso > 0.8:
                continue  # skip jets that are actually just single leptons

        min_distance = 0.4
        matched_particle = None
        for p in filter_particles(particles, [5, 4], False):
            distance = jet.delta_R(fj.PseudoJet(p.px(), p.py(), p.pz(), p.e()))
            if distance < min_distance:
                min_distance = distance
                matched_particle = p
        if matched_particle is None:
            for p in filter_particles(particles, [1, 2, 3, 21], False):
                part = fj.PseudoJet(p.px(), p.py(), p.pz(), p.e())
                if part.pt() > 5:
                    distance = (
                        jet.delta_R(fj.PseudoJet(p.px(), p.py(), p.pz(), p.e()))
                        + abs(part.pt() - jet.pt()) / jet.pt()
                    )
                    if distance < min_distance:
                        min_distance = distance
                        matched_particle = p

        jet_flavour = matched_particle.id() if matched_particle else 0  #        // 1000
        jet.flavour = jet_flavour
        matched_jets.append(jet)
    return matched_jets


attrExtractors = [
    # GEN LEVEL
    lambda j: j.pt(),  # 0
    lambda j: j.eta(),  # 1
    lambda j: j.phi(),  # 2
    lambda j: j.m(),  # 3
    lambda j: j.flavour,  # 4
    # RECO LEVEL
    lambda j: j.btag,  # 5
    lambda j: j.recoPt,  # 6
    lambda j: j.recoPhi,  # 7
    lambda j: j.recoEta,  # 8
    lambda j: j.muonsInJet[0].pT() if len(j.muonsInJet) > 0 else -1,  # 9
    lambda j: j.recoNConstituents,  # 10
    lambda j: j.nef,  # 11
    lambda j: j.nhf,  # 12
    lambda j: j.cef,  # 13
    lambda j: j.chf,  # 14
    lambda j: j.qgl,  # 15
    lambda j: j.jetId,  # 16
    lambda j: j.ncharged,  # 17
    lambda j: j.nneutral,  # 18
    lambda j: j.ctag,  # 19
    lambda j: j.nSV,  # 20
    lambda j: j.recoMass,  # 21
]
jet_dataset = np.empty((0, len(attrExtractors) + 1), float)  # + 1 for event number
nevents = 1000000
import sys

# if len(sys.argv) > 1:
#     nevents = int(sys.argv[1])
nevents = args.nevents
# Main loop
for i_event in range(nevents):
    if not pythia.next():
        continue

    # Create pseudojet inputs from stable particles
    particles = pythia.event
    #    pseudojets = [fj.PseudoJet(p.px(), p.py(), p.pz(), p.e()) for p in particles if is_stable(p)]
    pseudojets = []
    for i, p in enumerate(particles):
        if is_stable(p):
            pseudojet = fj.PseudoJet(p.px(), p.py(), p.pz(), p.e())
            pseudojet.set_user_index(i)
            pseudojets.append(pseudojet)

    # Cluster stable particles with FastJet
    clustered_sequence = fj.ClusterSequence(pseudojets, jet_def)
    clustered_jets = [j for j in clustered_sequence.inclusive_jets() if j.pt() > 15]

    for j in clustered_jets:
        j.pythiaConstituents = [particles[c.user_index()] for c in j.constituents()]
        j.muonsInJet = [x for x in j.pythiaConstituents if x.idAbs() == 13]
        j.elesInJet = [x for x in j.pythiaConstituents if x.idAbs() == 11]
        j.nef = (
            sum(
                [x.p() for x in j.pythiaConstituents if x.idAbs() in [111, 22]],
                pythia8.Vec4(0, 0, 0, 0),
            ).e()
            / j.e()
        )
        j.cef = (
            sum(
                [x.p() for x in j.pythiaConstituents if x.idAbs() in [13, -13]],
                pythia8.Vec4(0, 0, 0, 0),
            ).e()
            / j.e()
        )
        j.chf = (
            sum(
                [x.p() for x in j.pythiaConstituents if x.idAbs() in [211, -211]],
                pythia8.Vec4(0, 0, 0, 0),
            ).e()
            / j.e()
        )
        j.nhf = 1.0 - j.nef - j.cef - j.chf

    # Create collections of final muons and electrons
    final_muons = filter_particles(particles, [13])
    final_electrons = filter_particles(particles, [11])

    # Match clustered jets to Pythia original particles and determine the jet flavor
    matched_jets = matchAndClean(clustered_jets, particles)

    matched_jets[:] = map(addJetTagging, matched_jets)
    matched_jets[:] = map(addJetRecoKinematics, matched_jets)
    if i_event % 100 == 0:
        print("Event", i_event, len(jet_dataset))

    # Print the results
    if i_event < 2:
        print("Event", i_event)
        print("Final muons:", len(final_muons))
        print(
            "Final electrons:", len(final_electrons), [e.pT() for e in final_electrons]
        )
        print(
            "Matched jets and their flavors:",
            [
                (j.pt(), j.flavour, j.btag, j.px(), j.py(), j.pz())
                for j in matched_jets
                if j.pt() > 20
            ],
        )
    for jet in matched_jets:
        attributes = np.array([[att(jet) for att in attrExtractors]])
        attributes = np.append(attributes, [[i_event]], axis=1)  # add event number
        jet_dataset = np.vstack((jet_dataset, attributes))
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 180
# print(jet_dataset[:30,(0,6,4,5,9)])
# print(jet_dataset[jet_dataset[:,9]>0])
print(jet_dataset[:30, (0, 6, 4, 5, 9, 11, 12, 13, 14, 15, 16, 17, 18, 10)])

np.save(f"{args.output_path}", jet_dataset)
# End Pythia
pythia.stat()
