import string
import numpy as np
import os
import datetime   as dt


# WEB, DATABASE &  API 
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy # Para o Banco de Dados
from sqlalchemy import Column, Float, Integer, String, DateTime # Para manipulação das tabelas
from flask_marshmallow import Marshmallow  # PAra trabalhar junto com o SQLAlchemy
from flask_jwt_extended import JWTManager, jwt_required, create_access_token  # PAra tocken
import sqlite3

from xgboost import XGBRegressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Importa funções auxiliares e específicas de Machine Learning
import functions_ML  as ml



#-------------------------------------------------------------------------------------------------
# VARIAVEIS
#-------------------------------------------------------------------------------------------------
# Lista de nomes de campos das features PCA
lstFeaturesPCA = ['ESTADO - UF','RECEITA','k COMPONENTS','METODO','MODELO','RMSE','R2']
lstReceitasFeaturesPCA = ['INDICE','RANK','FEATURE','IMPORTANCE','SCORE']
lstRandomForest = ['ID', 'RECEITA','nTREE','nEXENOS','DURAÇÂO','RMSE']

lstReceitasFeatures = ['INDICE','FEATURE','IMPORTANCE','PREDICTION','TARGET']
#------

#-------------------------------------------------------------------------------------------------
# APP
#-------------------------------------------------------------------------------------------------
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')

# App configs
db = SQLAlchemy(app)
ma = Marshmallow(app)

# Conexão para trabalhar com comandos SQL
# Connecting to DB
# Abre o banco
connection, cursor = ml.checaArquivo('database.db')

#-------------------------------------------------------------------------------------------------
# COMONDOS CLI - para manutenção do banco em desenvolvimento, para usar: FLASK <nome_do_comando>
#-------------------------------------------------------------------------------------------------

# CRIAR BANCO VAZIO: > Flask db_create
@app.cli.command('db_create')
def db_create():
    db.create_all()
    print('Database criado com sucesso!')

# APAGAR BANCO: > Flask db_drop
@app.cli.command('db_drop')
def db_drop():
    db.drop_all()
    print('Database apagado com sucesso!')




#-------------------------------------------------------------------------------------------------
# ROUTEs 
#-------------------------------------------------------------------------------------------------

# PCA

# Rank receitas PCA json
@app.route('/ranks/pca/<string:estado>', methods=['GET'])
def pca(estado: str):
    
    # Executando a pesquisa
    sql = "select uf,receita,k,metodo,model,RMSE,R2 from Rank where uf ='" + estado  + "' AND metodo = 'PCA' order by RMSE  ASC"
    reg_count = cursor.execute(sql).fetchall()
    return jsonify(reg_count), 200


# Rank receitas PCA json
@app.route('/ranks/pca/html/<string:estado>', methods=['GET'])
def pcahtml(estado: str):


    # Rtorno da função
    retorno = ''
    cod = 200
    
    # Executando a pesquisa
    sql = "select uf,receita,k,metodo,model,RMSE,R2 from Rank where uf ='" + estado  + "' AND metodo = 'PCA' order by RMSE  ASC"
    reg_count = cursor.execute(sql).fetchall()
    if len(reg_count) >= 0:

        # pega as colunas

        #mota retorno
        conteudo = "<h1>Rank geral do PCA para o estado do " + estado + "</h1></br>" 
        
        # montado tabela
        conteudo = conteudo + '<table style="width:100%">'
        
        #nome dascolunas
        conteudo = conteudo + '<tr>'
        # Varre as colunas
        for c in lstFeaturesPCA:
            conteudo = conteudo + '<td><strong>' + str(c)+'</strong></td>'
        #Fecha tr
        conteudo = conteudo + '</tr>'
        
        #Varrendo o sql
        for l in reg_count:

            conteudo = conteudo + '<tr>'
            # Varre as colunas
            for c in l:
                conteudo = conteudo + '<td>' + str(c)+'</td>'
            #Fecha tr
            conteudo = conteudo + '</tr>'
        #fecha tabela
        conteudo = conteudo + '</table>'
        
        # Cria a página de retorno
        retorno = ml.html(conteudo)

    else:
        cod = 500
        
    return retorno, cod



# Rank receitas PCA com detalhes em features
@app.route('/ranks/pca/features/html/<string:estado>', methods=['GET'])
def pcaFeatureshtml(estado: str):

    # Rtorno da função
    retorno = ''
    cod = 200
    
    # Executando a pesquisa
    sql = "select id,uf,receita,k,metodo,model,RMSE,R2 from Rank where uf ='" + estado  + "' AND metodo = 'PCA' order by RMSE  ASC"
    cursor = connection.cursor()
    reg_count = cursor.execute(sql).fetchall()

    lst = []
    # Adiciona os dados na lista para facilitar o subselect
    for t in reg_count:
        tmp = []
        # Varre as colunas
        for i in t:
            tmp.append( str(i) )
        lst.append(tmp) 
    

    if len(lst) > 0:

        # pega as colunas

        #mota retorno
        conteudo = "<h1>Rank Features PCA para o estado do " + estado + "</h1></br>" 
        
        # montado tabela
        conteudo = conteudo + '<table style="width:100%">'
        
        #nome dascolunas
        conteudo = conteudo + '<tr>'
        
        # Varre as colunas
        for c in lstFeaturesPCA:
            conteudo = conteudo + '<td><strong>' + str(c)+'</strong></td>'
        #Fecha tr
        conteudo = conteudo + '</tr>'
        
        #Varrendo o sql
        id_rank = '0'
        for l in lst:
            # Pega o primeiro registro como id do rank e o deixa fora da tabela
            conteudo = conteudo + '<tr>'
            # Varre as colunas
            cont = 0
            for c in l:
                # Pega o id do rank
                if cont == 0:
                    id_rank = str(c)
                else:
                    # Imprime os campos
                    conteudo = conteudo + '<td>' + str(c)+'</td>'
                cont = cont +  1
            #Fecha tr
            conteudo = conteudo + '</tr>'

            # Monta subtabela com features
            # Executando a pesquisa
            sql = "select indice,Rankeameto,feature,importance,score from FeaturesPCA where id_rank = " +  id_rank  + " order by indice"
            cursor = connection.cursor()
            reg_count = cursor.execute(sql).fetchall()
            if len(reg_count) >= 0:
                conteudo = conteudo + '<tr><td><table>'

                #Cabeçalho das coluanas
                lstReceitasFeaturesPCA
                conteudo = conteudo + '<tr>'
                # Varre as colunas
                for cab in lstReceitasFeaturesPCA:
                    conteudo = conteudo + '<td><strong>' + str(cab)+'</strong></td>'
                #Fecha tr
                conteudo = conteudo + '</tr>'
                # Varre as colunas
                for p in reg_count:

                    conteudo = conteudo + '<tr>'
                    for x in p:
                        conteudo = conteudo + '<td>' + str(x)+'</td>'

                    conteudo = conteudo + '</tr>'
                conteudo = conteudo + '</table></td></tr></BR>'


                #Fecha td


            cont = cont + 1   

        #fecha tabela
        conteudo = conteudo + '</table>'
        
        # Cria a página de retorno
        retorno = ml.html(conteudo)

    else:
        cod = 500
        
    return retorno, cod


# Rank receitas PCA com detalhes em features
@app.route('/ranks/pca/features/html/<string:estado>/<string:receita>/', methods=['GET'])
def pcaFeaturesReceitahtml(estado: str,receita: str):

    # Rtorno da função
    retorno = ''
    cod = 200
    
    # Executando a pesquisa
    sql = "select id,uf,receita,k,metodo,model,RMSE,R2 from Rank where uf ='" + estado  + "' AND metodo = 'PCA'  AND receita = '" + receita  + "' order by RMSE  ASC"
    cursor = connection.cursor()
    reg_count = cursor.execute(sql).fetchall()

    lst = []
    # Adiciona os dados na lista para facilitar o subselect
    for t in reg_count:
        tmp = []
        # Varre as colunas
        for i in t:
            tmp.append( str(i) )
        lst.append(tmp) 
    

    if len(lst) > 0:

        # pega as colunas

        #mota retorno
        conteudo = "<h1>Rank Features PCA para o estado do " + estado + "</h1></br>" 
        
        # montado tabela
        conteudo = conteudo + '<table style="width:100%">'
        
        #nome dascolunas
        conteudo = conteudo + '<tr>'
        
        # Varre as colunas
        for c in lstFeaturesPCA:
            conteudo = conteudo + '<td><strong>' + str(c)+'</strong></td>'
        #Fecha tr
        conteudo = conteudo + '</tr>'
        
        #Varrendo o sql
        id_rank = '0'
        for l in lst:
            # Pega o primeiro registro como id do rank e o deixa fora da tabela
            conteudo = conteudo + '<tr>'
            # Varre as colunas
            cont = 0
            for c in l:
                # Pega o id do rank
                if cont == 0:
                    id_rank = str(c)
                else:
                    # Imprime os campos
                    conteudo = conteudo + '<td>' + str(c)+'</td>'
                cont = cont +  1
            #Fecha tr
            conteudo = conteudo + '</tr>'

            # Monta subtabela com features
            # Executando a pesquisa
            sql = "select indice,Rankeameto,feature,importance,score from FeaturesPCA where id_rank = " +  id_rank  + " order by indice"
            cursor = connection.cursor()
            reg_count = cursor.execute(sql).fetchall()
            if len(reg_count) >= 0:
                conteudo = conteudo + '<tr><td><table>'

                #Cabeçalho das coluanas
                lstReceitasFeaturesPCA
                conteudo = conteudo + '<tr>'
                # Varre as colunas
                for cab in lstReceitasFeaturesPCA:
                    conteudo = conteudo + '<td><strong>' + str(cab)+'</strong></td>'
                #Fecha tr
                conteudo = conteudo + '</tr>'
                # Varre as colunas
                for p in reg_count:

                    conteudo = conteudo + '<tr>'
                    for x in p:
                        conteudo = conteudo + '<td>' + str(x)+'</td>'

                    conteudo = conteudo + '</tr>'
                conteudo = conteudo + '</table></td></tr></BR>'


                #Fecha td


            cont = cont + 1   

        #fecha tabela
        conteudo = conteudo + '</table>'
        
        # Cria a página de retorno
        retorno = ml.html(conteudo)

    else:
        cod = 500
        
    return retorno, cod



# Rank receitas PCA json
@app.route('/standalone/randomforest/training/htmml', methods=['GET'])
def randomForest():


    # Rtorno da função
    retorno = ''
    cod = 200
    
    # Executando a pesquisa
    sql = "select id, receita,ntree,nexenos,duracao,RMSE from RandomForest order by id"
    reg_count = cursor.execute(sql).fetchall()
    if len(reg_count) >= 0:

        # pega as colunas

        #mota retorno
        conteudo = "<h1>Grid Search do modelo stand alone Random Forest</h1></br>" 
        
        # montado tabela
        conteudo = conteudo + '<table style="width:100%">'
        
        #nome dascolunas
        conteudo = conteudo + '<tr>'
        # Varre as colunas
        for c in lstRandomForest:
            conteudo = conteudo + '<td><strong>' + str(c)+'</strong></td>'
        #Fecha tr
        conteudo = conteudo + '</tr>'
        
        #Varrendo o sql
        for l in reg_count:

            conteudo = conteudo + '<tr>'
            # Varre as colunas
            for c in l:
                conteudo = conteudo + '<td>' + str(c)+'</td>'
            #Fecha tr
            conteudo = conteudo + '</tr>'
        #fecha tabela
        conteudo = conteudo + '</table>'
        
        # Cria a página de retorno
        retorno = ml.html(conteudo)

    else:
        cod = 500
        
    return retorno, cod



# Rank receitas PCA com detalhes em features
@app.route('/standalone/xboost/features/html/<string:estado>', methods=['GET'])
def StandAloeXBoostFeaturesReceitahtml(estado: str):

    # Rtorno da função
    retorno = ''
    cod = 200
    
    # Executando a pesquisa
    sql = "select id,uf,receita,k,metodo,model,RMSE,R2 from Rank where uf ='" + estado  + "' AND metodo = 'STANDALONE' AND model = 'XBOOST'  order by RMSE  ASC"
    cursor = connection.cursor()
    reg_count = cursor.execute(sql).fetchall()

    lst = []
    # Adiciona os dados na lista para facilitar o subselect
    for t in reg_count:
        tmp = []
        # Varre as colunas
        for i in t:
            tmp.append( str(i) )
        lst.append(tmp) 

    if len(lst) > 0:

        # pega as colunas

        #mota retorno
        conteudo = "<h1>Rank Features Stand Alone (XBOST) para o estado do " + estado + "</h1>" 
        
        # montado tabela
        conteudo = conteudo + '<table style="width:70%">'
        
        #nome dascolunas
        conteudo = conteudo + '<tr>'
        
        # Varre as colunas
        for c in lstFeaturesPCA:
            conteudo = conteudo + '<td><strong>' + str(c)+'</strong></td>'
        #Fecha tr
        conteudo = conteudo + '</tr>'
        
        #Varrendo o sql
        id_rank = '0'
        for l in lst:
            # Pega o primeiro registro como id do rank e o deixa fora da tabela
            conteudo = conteudo + '<tr>'
            # Varre as colunas
            cont = 0
            for c in l:
                # Pega o id do rank
                if cont == 0:
                    id_rank = str(c)
                else:
                    # Imprime os campos
                    conteudo = conteudo + '<td>' + str(c)+'</td>'
                cont = cont +  1
            #Fecha tr
            conteudo = conteudo + '</tr>'

            # Monta subtabela com features
            # Executando a pesquisa
            
            sql = "select indice,feature,importance,prediction,target from Features where id_rank = " +  id_rank  + " order by indice"
            cursor = connection.cursor()
            reg_count = cursor.execute(sql).fetchall()
            if len(reg_count) >= 0:
                conteudo = conteudo + '<tr><td><table>'

                #Cabeçalho das coluanas
                lstReceitasFeaturesPCA
                conteudo = conteudo + '<tr>'
                # Varre as colunas
                for cab in lstReceitasFeatures:
                    conteudo = conteudo + '<td><strong>' + str(cab)+'</strong></td>'
                #Fecha tr
                conteudo = conteudo + '</tr>'
                # Varre as colunas
                for p in reg_count:

                    conteudo = conteudo + '<tr>'
                    for x in p:
                        conteudo = conteudo + '<td>' + str(x)+'</td>'

                    conteudo = conteudo + '</tr>'
                conteudo = conteudo + '</table></td></tr></BR>'


                #Fecha td


            cont = cont + 1   

        #fecha tabela
        conteudo = conteudo + '</table>'
        
        # Cria a página de retorno
        retorno = ml.html(conteudo)

    else:
        cod = 500
        
    return retorno, cod





# Rank receitas PCA com detalhes em features
@app.route('/standalone/randomforest/features/html/<string:estado>', methods=['GET'])
def StandAloeFeaturesReceitahtml(estado: str):

    # Rtorno da função
    retorno = ''
    cod = 200
    
    # Executando a pesquisa
    sql = "select id,uf,receita,k,metodo,model,RMSE,R2 from Rank where uf ='" + estado  + "' AND metodo = 'STANDALONE'  AND model = 'RANDOMFOREST'   order by RMSE  ASC"
    cursor = connection.cursor()
    reg_count = cursor.execute(sql).fetchall()

    lst = []
    # Adiciona os dados na lista para facilitar o subselect
    for t in reg_count:
        tmp = []
        # Varre as colunas
        for i in t:
            tmp.append( str(i) )
        lst.append(tmp) 

    if len(lst) > 0:

        # pega as colunas

        #mota retorno
        conteudo = "<h1>Rank Features Stand Alone para o estado do " + estado + "</h1></br>" 
        
        # montado tabela
        conteudo = conteudo + '<table style="width:100%">'
        
        #nome dascolunas
        conteudo = conteudo + '<tr>'
        
        # Varre as colunas
        for c in lstFeaturesPCA:
            conteudo = conteudo + '<td><strong>' + str(c)+'</strong></td>'
        #Fecha tr
        conteudo = conteudo + '</tr>'
        
        #Varrendo o sql
        id_rank = '0'
        for l in lst:
            # Pega o primeiro registro como id do rank e o deixa fora da tabela
            conteudo = conteudo + '<tr>'
            # Varre as colunas
            cont = 0
            for c in l:
                # Pega o id do rank
                if cont == 0:
                    id_rank = str(c)
                else:
                    # Imprime os campos
                    conteudo = conteudo + '<td>' + str(c)+'</td>'
                cont = cont +  1
            #Fecha tr
            conteudo = conteudo + '</tr>'

            # Monta subtabela com features
            # Executando a pesquisa
            
            sql = "select indice,feature,importance,prediction,target from Features where id_rank = " +  id_rank  + " order by indice"
            cursor = connection.cursor()
            reg_count = cursor.execute(sql).fetchall()
            if len(reg_count) >= 0:
                conteudo = conteudo + '<tr><td><table>'

                #Cabeçalho das coluanas
                lstReceitasFeatures
                conteudo = conteudo + '<tr>'
                # Varre as colunas
                for cab in lstReceitasFeatures:
                    conteudo = conteudo + '<td><strong>' + str(cab)+'</strong></td>'
                #Fecha tr
                conteudo = conteudo + '</tr>'
                # Varre as colunas
                for p in reg_count:

                    conteudo = conteudo + '<tr>'
                    for x in p:
                        conteudo = conteudo + '<td>' + str(x)+'</td>'

                    conteudo = conteudo + '</tr>'
                conteudo = conteudo + '</table></td></tr></BR>'


                #Fecha td


            cont = cont + 1   

        #fecha tabela
        conteudo = conteudo + '</table>'
        
        # Cria a página de retorno
        retorno = ml.html(conteudo)

    else:
        cod = 500
        
    return retorno, cod








#----------------------------------------------------------------------------------------------------------------------------------------------------
# DB MODELS
#----------------------------------------------------------------------------------------------------------------------------------------------------

# DB Models
# database models


# CAPITAIS
class Capital(db.Model):
    __tablename__ = 'Capitals'
    uf = Column(String, primary_key=True)
    nome = Column(String)


class CapitalSchema(ma.Schema):
    class Meta:
        fields = ('uf', 'nome')

capital_schema = CapitalSchema()
capitals_schema = CapitalSchema(many=True)


# RECEITAS
class Receitas(db.Model):
    __tablename__ = 'Receitas'
    id = Column(String,  primary_key=True)
    nome = Column(String,  primary_key=True)

class ReceitaSchema(ma.Schema):
    class Meta:
        fields = ('id','nome')

receita_schema = ReceitaSchema()
receitas_schema = ReceitaSchema(many=True)

# FEATURES RANK
class Rank(db.Model):
    __tablename__ = 'Rank'
    id          = Column(Integer,  db.Sequence('seq_reg_id', start=1, increment=1), primary_key=True)
    uf          = Column(String)
    receita     = Column(String)
    k           = Column(Integer)
    metodo      = Column(String)
    model       = Column(String)
    nrArvores   = Column(Float)
    qntAmosNo   = Column(Float)
    MAE         = Column(Float)
    MSE         = Column(Float)
    RMSE        = Column(Float)
    R2          = Column(Float)

class RankSchema(ma.Schema):
    class Meta:
        fields = ('id','uf','receita','k','metodo','model','nrArvores','qntAmosNo','MAE','MSE','RMSE','R2')
    
rank_schema = RankSchema()
ranks_schema = RankSchema(many=True)


# FEATURES FEATURES
class FeaturesPCA(db.Model):
    __tablename__   = 'FeaturesPCA'
    id              = Column(Integer,  primary_key=True)
    id_rank         = Column(Integer)
    indice          = Column(Integer)
    Rankeameto      = Column(String)
    feature         = Column(String)
    Importance      = Column(Float)
    score           = Column(String)

class FeaturesPCA(ma.Schema):
    class Meta:
        fields = ('id','id_rank','indice','Rankeameto','feature','importance','score')
    
featurespca_schema = FeaturesPCA()
featurespcas_schema = FeaturesPCA(many=True)

# FEATURES FEATURES
class Features(db.Model):
    __tablename__   = 'Features'
    id              = Column(Integer,  primary_key=True)
    id_rank         = Column(Integer)
    indice          = Column(Integer)
    feature        = Column(String)
    importance      = Column(Float)


class Features(ma.Schema):
    class Meta:
        fields = ('id','id_rank','indice','feature','importance')
    
features_schema = Features()
featuress_schema = Features(many=True)


# PREDICAO FEATURES
class Predicao(db.Model):
    __tablename__   = 'Predicao'
    id              = Column(Integer,  primary_key=True)
    id_rank         = Column(Integer)
    indice          = Column(Integer)
    predicao        = Column(Float)
    target          = Column(Float)


class Predicao(ma.Schema):
    class Meta:
        fields = ('id','id_rank','indice','prediction','target')
    
predicao_schema = Predicao()
predicaos_schema = Predicao(many=True)


# FRANDOM FOREST
class RandomForest(db.Model):
    __tablename__   = 'RandomForest'
    id              = Column(Integer,  primary_key=True)
    receita         = Column(String)
    ntree           = Column(Integer)
    nexenos         = Column(Integer)
    duracao         = Column(Float)
    RMSE            = Column(Float)

    
class RandomForest(ma.Schema):
    class Meta:
        fields = ('id', 'receita', 'ntree', 'nexenos', 'duracao', 'RMSE')
    
randomforest_schema = RandomForest()
randomforests_schema = RandomForest(many=True)





#-------------------------------------------------------------------------------------------------
# FUNÇÂO MAIN
#-------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
