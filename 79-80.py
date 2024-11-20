import pandas as pd
from chemdataextractor.doc import Document
import json
import re
from chemdataextractor.model.model import Compound
from chemdataextractor.model.base import BaseModel, StringType, ModelType
from chemdataextractor.model.units.ratio import RatioModel
from chemdataextractor.parse.elements import I, W, Any, R
from chemdataextractor.parse.actions import join
#from chemdataextractor.reader.plaintext import PlainTextReader
from chemdataextractor.parse.auto import AutoSentenceParser,AutoSentenceParserOptionalCompound


# Define model
common_Latticeform1 = (
    I('single') + I('crystal')
).add_action(join)

common_Latticeform2 = (
    I('polycrystalline')
).add_action(join)

space_group = (
    W('P1') | W('P-1') | W('P2') | W('P21') | W('C2') | W('Pm') | W('Pc') |
    W('Cm') | W('Cc') | W('P2/m') | W('P21/m') | W('C2/m') | W('P2/c') |
    W('P21/c') | W('C2/c') | W('P222') | W('P2221') | W('P21212') | W('P212121') |
    W('C222') | W('C2221') | W('F222') | W('I222') | W('I212121') | W('Pmm2') |
    W('Pmc21') | W('Pcc2') | W('Pma2') | W('Pca21') | W('Pnc2') | W('Pmn21') |
    W('Pba2') | W('Pna21') | W('Pnn2') | W('Cmm2') | W('Cmc21') | W('Ccc2') |
    W('Amm2') | W('Abm2') | W('Ama2') | W('Aba2') | W('Fmm2') | W('Fdd2') |
    W('Imm2') | W('Iba2') | W('Ima2') | W('Pmmm') | W('Pnnn') | W('Pccm') |
    W('Pban') | W('Pmma') | W('Pnna') | W('Pmna') | W('Pcca') | W('Pbam') |
    W('Pccn') | W('Pbcm') | W('Pnnm') | W('Pmmn') | W('Pbcn') | W('Pbca') |
    W('Pnma') | W('Cmmm') | W('Cccm') | W('Cmma') | W('Ccca') | W('Fmmm') |
    W('Fddd') | W('Immm') | W('Ibam') | W('Ibca') | W('Imma') | W('P4') |
    W('P41') | W('P42') | W('P43') | W('I4') | W('I41') | W('P-4') | W('I-4') |
    W('P4/m') | W('P42/m') | W('P4/n') | W('P42/n') | W('I4/m') | W('I41/a') |
    W('P422') | W('P4212') | W('P4122') | W('P41212') | W('P4222') | W('P42212') |
    W('P4322') | W('P43212') | W('I422') | W('I4122') | W('P4mm') | W('P4bm') |
    W('P42cm') | W('P42nm') | W('P4cc') | W('P4nc') | W('P42mc') | W('P42bc') |
    W('I4mm') | W('I4cm') | W('I41md') | W('I41cd') | W('P-42m') | W('P-42c') |
    W('P-421m') | W('P-421c') | W('P-4m2') | W('P-4c2') | W('P-4b2') | W('P-4n2') |
    W('I-4m2') | W('I-4c2') | W('I-42m') | W('I-42d') | W('P4/mmm') | W('P4/mcc') |
    W('P4/nbm') | W('P4/nnc') | W('P4/mbm') | W('P4/mnc') | W('P4/nmm') | W('P4/ncc') |
    W('P42/mmc') | W('P42/mcm') | W('P42/nbc') | W('P42/nnm') | W('P42/mbc') |
    W('P42/mnm') | W('P42/nmc') | W('P42/ncm') | W('I4/mmm') | W('I4/mcm') |
    W('I41/amd') | W('I41/acd') | W('P3') | W('P31') | W('P32') | W('R3') |
    W('P-3') | W('R-3') | W('P312') | W('P321') | W('P3112') | W('P3121') |
    W('P3212') | W('P3221') | W('R32') | W('P3m1') | W('P31m') | W('P3c1') |
    W('P31c') | W('R3m') | W('R3c') | W('P-31m') | W('P-31c') | W('P-3m1') |
    W('P-3c1') | W('R-3m') | W('R-3c') | W('P6') | W('P61') | W('P65') | W('P62') |
    W('P64') | W('P63') | W('P-6') | W('P6/m') | W('P63/m') | W('P622') |
    W('P6122') | W('P6522') | W('P6222') | W('P6422') | W('P6322') | W('P6mm') |
    W('P6cc') | W('P63cm') | W('P63mc') | W('P-6m2') | W('P-6c2') | W('P-62m') |
    W('P-62c') | W('P6/mmm') | W('P6/mcc') | W('P63/mcm') | W('P63/mmc') |
    W('P23') | W('F23') | W('I23') | W('P213') | W('I213') | W('Pm-3') |
    W('Pn-3') | W('Fm-3') | W('Fd-3') | W('Im-3') | W('Pa-3') | W('Ia-3') |
    W('P432') | W('P4232') | W('F432') | W('F4132') | W('I432') | W('P4332') |
    W('P4132') | W('I4132') | W('P-43m') | W('F-43m') | W('I-43m') | W('P-43n') |
    W('F-43c') | W('I-43d') | W('Pm-3m') | W('Pn-3n') | W('Pm-3n') | W('Pn-3m') |
    W('Fm-3m') | W('Fm-3c') | W('Fd-3m') | W('Fd-3c') | W('Im-3m') | W('Ia-3d')
).add_action(join)

Lattice_Parameter = (R('a\s*=\s*\d+(\.\d+)?\s*(Å|nm|pm|μm)(,?\s*b\s*=\s*\d+(\.\d+)?\s*(Å|nm|pm|μm))?(,?\s*c\s*=\s*\d+(\.\d+)?\s*(Å|nm|pm|μm))?,?\s*(α\s*=\s*\d+(\.\d+)?\s*°)?(,?\s*β\s*=\s*\d+(\.\d+)?\s*°)?(,?\s*γ\s*=\s*\d+(\.\d+)?\s*°)?', re.I)).add_action(join)
class singlecrystal(BaseModel):
    specifier_expr = (I('single') + I('crystal')).add_action(join)
    specifier = StringType(parse_expression=specifier_expr, required=True, contextual=True)
    #compound = ModelType(Compound)
    raw_value = StringType(parse_expression=(common_Latticeform1).add_action(join), required=True, contextual=True)
    parsers = [AutoSentenceParserOptionalCompound()]
class polycrystalline(BaseModel):
    specifier_expr = (I('polycrystalline')).add_action(join)
    specifier = StringType(parse_expression=specifier_expr, required=True, contextual=True)
    #compound = ModelType(Compound)
    raw_value = StringType(parse_expression=(common_Latticeform2).add_action(join), required=True, contextual=True)
    parsers = [AutoSentenceParserOptionalCompound()]
class SpaceGroup(BaseModel):
    specifier_expr = (I('space') + I('group')).add_action(join)
    specifier = StringType(parse_expression=specifier_expr, required=True, contextual=True)
    raw_value = StringType(parse_expression=(space_group).add_action(join), required=True, contextual=True)
    #compound = ModelType(Compound)
    parsers = [AutoSentenceParserOptionalCompound()]
class LatticeParameter(BaseModel):
    specifier_expr = (I('Lattice') | I('Lattice') + I('parameter')).add_action(join)
    specifier = StringType(parse_expression=specifier_expr, required=True, contextual=True)
    raw_value = StringType(parse_expression=(Lattice_Parameter).add_action(join), required=True, contextual=True)
    #parsers = [AutoSentenceParser()]
    parsers = [AutoSentenceParserOptionalCompound()]
class TensileStrain(RatioModel):
    specifier_expr = (I('ε') | I('εt') | I('Tensile') | I('Tensile') + I('strain') | I('Tension-induced') + I('strain') | I('Strain') + I('under') + I('Tension') | I('Axial') + I('Tensile') + I('strain') | I('Strain') + I('due') + I('to') + I('Tensile') + I('Loading') | I('Tensile') + I('Deformation') | I('Tensile') + I('Stress-Strain') + I('Relationship') | I('Longitudinal') + I('strain') | I('Tension-related') + I('strain')).add_action(join)
    specifier = StringType(parse_expression=specifier_expr, required=True, contextual=True)
    #compound = ModelType(Compound)
    parsers = [AutoSentenceParserOptionalCompound()]
class CompressStrain(RatioModel):
    specifier_expr = (I('ε')+ I('c')| I('εc') | I('Compress') | I('Compress') + I('strain') | I('Compressive') + I('strain') | I('Compression-induced') + I('strain') | I('Strain') + I('under') + I('Compression') | I('Axial') + I('Compressive') + I('strain') | I('Compressive') + I('Deformation') | I('Compressive') + I('Stress-Strain') + I('Relationship') | I('Bulk') + I('Strain') | I('Compression-related') + I('strain')).add_action(join)
    specifier = StringType(parse_expression=specifier_expr, required=True, contextual=True)
    #compound = ModelType(Compound)
    parsers = [AutoSentenceParserOptionalCompound()]
class BendingStrain(RatioModel):
    specifier_expr = (I('ε')+ I('c') | I('εb') | I('𝜀') | I('bending') | I('bending') + I('strain') | I('Bending') + I('Curvature') | I('Strain') + I('due') + I('to') + I('Curvature') | I('Flexural') + I('Deformation') | I('Curvature-induced') + I('Strain') | I('Bending-induced') + I('Strain')).add_action(join)
    specifier = StringType(parse_expression=specifier_expr, required=True, contextual=True)
    #compound = ModelType(Compound)
    #parsers = [AutoSentenceParser()]
    parsers = [AutoSentenceParserOptionalCompound()]
class Ceramics(BaseModel):
    specifier = StringType(parse_expression=Any(), required=False, contextual=False)
    compound = ModelType(Compound, required=True, contextual=True)
    SC = ModelType(singlecrystal, required=False, contextual=True)
    PC = ModelType(polycrystalline, required=False, contextual=True)
    SG = ModelType(SpaceGroup, required=False, contextual=True)
    LP = ModelType(LatticeParameter, required=False, contextual=True)
    TS = ModelType(TensileStrain, required=False, contextual=False)
    C = ModelType(CompressStrain, required=False, contextual=False)
    BS = ModelType(BendingStrain, required=False, contextual=False)
    parsers = [AutoSentenceParserOptionalCompound()]

# Loop through the years from 1960 to 2024
for year in range(1979, 1981):
    # Generate the filename based on the year
    input_filename = f"{year}.json"
    output_filename = f"output{year}.xlsx"

    try:
        # Load the JSON file
        with open(input_filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Prepare the rows for the DataFrame
        rows = []

        # Process each record in the JSON data
        for record in json_data:
            doi = record.get("doi", "none")  # Get DOI
            original_text = record.get("content", {}).get("full-text-retrieval-response", {}).get("originalText", "none")
            # Create a Document object using the original text
            doc = Document(str(original_text))
            doc.models = [Ceramics]

            # Serialize the records into a structured form
            records_data = [record.serialize() for record in doc.records]

            # Extract relevant data from the records
            for record in records_data:
                ceramics = record.get('Ceramics', {})
                compound_data = ceramics.get('compound', {}).get('Compound', {})
                compound_name = compound_data.get('names', ['none'])[0]  # Use the first name if exists
                single_crystal = ceramics.get('SC', {}).get('singlecrystal', {}).get('raw_value', 'none')
                poly_crystalline = ceramics.get('PC', {}).get('polycrystalline', {}).get('raw_value', 'none')
                space_group = ceramics.get('SG', {}).get('SpaceGroup', {}).get('raw_value', 'none')
                lattice_parameter = ceramics.get('LP', {}).get('LatticeParameter', {}).get('raw_value', 'none')
                tensile_strain = ceramics.get('TS', {}).get('TensileStrain', {}).get('raw_value', 'none')
                compress_strain = ceramics.get('C', {}).get('CompressStrain', {}).get('raw_value', 'none')
                bending_strain = ceramics.get('BS', {}).get('BendingStrain', {}).get('raw_value', 'none')

                # Skip records where all strains are 'none'
                if tensile_strain == 'none' and compress_strain == 'none' and bending_strain == 'none':
                    continue

                # Add a row with DOI and extracted information
                rows.append([
                    doi, compound_name, single_crystal, poly_crystalline,
                    space_group, lattice_parameter, tensile_strain, compress_strain, bending_strain
                ])

        # Convert the rows into a DataFrame
        if rows:  # Only create a file if there are rows to write
            df = pd.DataFrame(rows, columns=[
                "DOI", "Compound Name", "Single Crystal", "Polycrystalline",
                "Space Group", "Lattice Parameter", "Tensile Strain",
                "Compress Strain", "Bending Strain"
            ])
            # Export the DataFrame to an Excel file
            df.to_excel(output_filename, index=False)
            print(f"Successfully processed {input_filename} -> {output_filename}")
        else:
            print(f"No valid records found in {input_filename}, skipping file generation.")

    except FileNotFoundError:
        print(f"File {input_filename} not found. Skipping this year.")
    except Exception as e:
        print(f"Error processing {input_filename}: {e}")