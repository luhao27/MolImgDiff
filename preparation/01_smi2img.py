
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import os    
from rdkit.Chem import Descriptors
import json

# 1_qed_unipriID.png
def read_xyz():
    import os
    # define file path
    file_path = '../datasets/chembel_512/smi_new_512_uni.smi'
    save_path = '../datasets/chembel_512/chembel_pic'
    pro_path = '../datasets/chembel_512/pdb_new_512_uni.smi'
  

    # target array  
    pdb_arr = open(pro_path, 'r').readlines()

    
    total= 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles)
            # 获取分子的QED属性
            qed = Descriptors.qed(mol)
            # 取qed的整数部分
            qed = int(qed * 100)
            
            # 获取pdb的名字
            pro_id = pdb_arr[total].strip()
            # 拼接成为图片名字
            save_file = os.path.join(save_path, str(total) + "_" + str(qed) + "_" + pro_id)
            
            svg2png(smiles=smiles, save_path=save_file)
            total += 1
            if total % 10000 == 0:
                print(total)
                # quit()

    print("all done: {}".format(total))    
    
        

def svg2png(smiles, save_path):
    """
    smiles: SMILES字符串
    save_path: 保存路径
    """
    import cairosvg
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from PIL import Image
    from io import BytesIO

    mol = Chem.MolFromSmiles(smiles)
    
    
    # 显示符号
    # for atom in mol.GetAtoms():
    #     atom_index = atom.GetIdx()
    #     atom_symbol = atom.GetSymbol()
    #     # print(atom_symbol)
    #     mol.GetAtomWithIdx(atom_index).SetProp("_displayLabel", atom_symbol)
    


    d = rdMolDraw2D.MolDraw2DSVG(256, 256)
    
    do = d.drawOptions()
    # 是否画出原子序号
    # do.addAtomIndices = True
    # 立体注释
    # do.addStereoAnnotation = True
    do.baseFontSize=0.9
    # do.explicitMethyl = True
    do.annotationFontScale = 0.9
    # QM9
    # do.bondLineWidth = 3
    # geom
    do.bondLineWidth = 2

    d.DrawMolecule(mol)
    d.FinishDrawing()
     
    svg_data = d.GetDrawingText()
    
    # svg_data = svg_data.replace('<svg', '<svg style="background-color: red;"')
    # 保存 SVG 图像到文件
    with open(save_path, 'w') as svg_file:
        svg_file.write(svg_data)
    
    
    cairosvg.svg2png(url=save_path, write_to=save_path + '.png', dpi=300)
    # 删除save_path
    os.remove(save_path)



if __name__ == '__main__':
    # demo2
    read_xyz()
    

